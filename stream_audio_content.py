import asyncio
import websockets
import soundfile as sf
from bidi.algorithm import get_display
import arabic_reshaper


WEBSOCKET_URL = "ws://localhost:8001/ws/transcribe_stream_v2/"

# Audio file to stream
AUDIO_FILE = "/Users/a1197/Downloads/projet_machine_learning/audio_data/audioPLus660.wav"

async def send_heartbeat(websocket):
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(5)
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed")
            break

async def stream_audio_to_websocket(file_path, websocket_url):
    """
    Stream an audio file to the WebSocket server.
    """
    try:
        # Open a WebSocket connection
        print("launching websocket connection...")
        async with websockets.connect(websocket_url) as websocket:
            print(f"Connected to WebSocket server: {websocket_url}")
            asyncio.create_task(send_heartbeat(websocket))
            # Open the audio file
            with sf.SoundFile(file_path, "rb") as audio_file:
                # Check file properties
                print(f"Streaming audio with sample rate: {audio_file.samplerate}, channels: {audio_file.channels}")
                
                # Stream the file in chunks
                chunk_size = 16000  # Number of frames to send per message
                while True:
                    # Read the next chunk of frames
                    audio_chunk = audio_file.read(chunk_size, dtype="int16")  # Change dtype as needed
                    print("audio_chunk.size",audio_chunk.size)
                    if not audio_chunk.size:
                        break  # End of file reached
                    # Send the audio chunk as binary data
                    await websocket.send(audio_chunk.tobytes())
                    # Optionally, receive a server response (if the server sends back responses)
                    response = await websocket.recv()

                    reshaped_text = arabic_reshaper.reshape(response)
                    print("Server response:", get_display(reshaped_text))
                print("Sending end of stream signal...")
                await websocket.send(b"end_of_stream")  # End of stream signal as bytes
                print("End of stream signal sent.")
            print("Finished streaming audio.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(stream_audio_to_websocket(AUDIO_FILE, WEBSOCKET_URL))