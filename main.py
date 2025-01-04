from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
import os
import soundfile as sf
from io import BytesIO
import websockets
from fastapi.responses import HTMLResponse
import numpy as np
import logging 
from lang_trans.arabic import buckwalter
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
from tools import transcribe_multiple_audio, transcribe_audio
import torchaudio

app = FastAPI()
# Load the processor and model
# "mohammed/wav2vec2-large-xlsr-arabic"
# "elgeish/wav2vec2-large-xlsr-53-arabic"
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir="models")
model = Wav2Vec2ForCTC.from_pretrained(model_name, force_download=False, cache_dir="models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device) 
# audio_files = os.listdir("./data")
# transcriptions = transcribe_multiple_audio(audio_files, processor, model, device )

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@app.get("/")
async def get():
    return HTMLResponse(html)

# Endpoint to transcribe a single audio file
@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as f:
        f.write(await file.read())

    transcription =  transcribe_audio(temp_file, processor, model, device )
    os.remove(temp_file)

    return {"file_name": file.filename, "transcription": transcription}

# Endpoint to process multiple audio files
@app.post("/transcribe_multiple/")
async def transcribe_multiple(files: list[UploadFile] = File(...)):
    transcriptions = {}

    # Save and process each file
    for file in files:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        transcription = transcribe_audio(temp_file, processor, model, device )
        transcriptions[file.filename] = transcription
        
        os.remove(temp_file)

    return {"transcriptions": transcriptions}


@app.websocket("/ws/transcribe_stream_v2/")
async def transcribe_stream(websocket: WebSocket):
    client_host = websocket.client.host
    print(f"Connection attempt from {client_host}")
    await websocket.accept()
    print(f"WebSocket connection accepted from {client_host}")
    buffer = bytearray()
    SAMPLE_RATE = 16000  # Define your sample rate
    CHUNK_SIZE = SAMPLE_RATE * 2 # 1 second of audio, 16-bit samples (2 bytes per sample)

    try:
        while True:
            # Receive audio data from the WebSocket client
            try:
                data = await websocket.receive_bytes()
                print("Received data size:", len(data))
                buffer.extend(data)

                # Process every 1 second of audio
                if len(buffer) >= CHUNK_SIZE :#CHUNK_SIZE:
                    # Convert the buffer to a waveform tensor
                    audio_samples = np.frombuffer(buffer[:CHUNK_SIZE], dtype=np.int16).astype(np.float32) / 32768.0
                    audio_tensor = torch.tensor([audio_samples]).to(device)  # Add batch dimension
                    buffer = buffer[CHUNK_SIZE:]  # Remove processed data

                    # Transcription logic
                    with torch.no_grad():
                        logits = model(audio_tensor).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)[0]
                    transcription = buckwalter.untrans(transcription)
                    # Send the transcription back to the client
                    await websocket.send_text(transcription)
                else:
                    print(f"len buffer {len(buffer)} is not sufficient {CHUNK_SIZE} )")

            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed by the client.")
                break  # Exit the loop if the connection is closed

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
        print("WebSocket connection closed.")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         print('eatata', data)
#         await websocket.send_text(f"Message text was: {data}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)



# uvicorn main:app --host 0.0.0.0 --port 8080 --reload --access-log