import logging
import os

import numpy as np
import torch
import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse
from lang_trans.arabic import buckwalter
from tools import transcribe_audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

app = FastAPI()
# Load the processor and model
# "mohammed/wav2vec2-large-xlsr-arabic"
# "elgeish/wav2vec2-large-xlsr-53-arabic"
model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir="models")
model = Wav2Vec2ForCTC.from_pretrained(model_name, force_download=False, cache_dir="models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

SAMPLE_RATE = 16000  # Define your sample rate
SEGMENT_LENGTH = 1  # 1 second of audio
CHUNK_SIZE = SAMPLE_RATE * SEGMENT_LENGTH * 2  # 16-bit samples (2 bytes per sample)


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

    transcription = transcribe_audio(temp_file, processor, model, device)
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

        transcription = transcribe_audio(temp_file, processor, model, device)
        transcriptions[file.filename] = transcription

        os.remove(temp_file)

    return {"transcriptions": transcriptions}


@app.websocket("/ws/transcribe_stream/")
async def transcribe_stream(websocket: WebSocket):
    client_host = websocket.client.host
    logger.info(f"Connection attempt from {client_host}")
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from {client_host}")
    buffer = bytearray()
    try:
        while True:
            # Receive audio data from the WebSocket client
            try:
                data = await websocket.receive_bytes()
                logger.info(f"Received data size: {len(data)}")
                if data == b"end_of_stream":
                    logger.info("Received end_of_stream signal.")
                    break  # Exit the loop when the "end_of_stream" signal is received

                buffer.extend(data)

                # Process every 1 second of audio
                if len(buffer) >= 0:
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
                    logger.info(f"len buffer {len(buffer)} is not sufficient {CHUNK_SIZE} ")

            except websockets.exceptions.ConnectionClosed:
                logger.error("WebSocket connection closed by the client.")
                break  # Exit the loop if the connection is closed

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            # Close the WebSocket connection at the end
            await websocket.close()
            logger.info("WebSocket connection closed.")
        except Exception as e:
            logger.error(f"Error closing WebSocket connection: {e}")


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
