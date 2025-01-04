import asyncio
import logging

import arabic_reshaper
import soundfile as sf
import websockets
from bidi.algorithm import get_display

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

WEBSOCKET_URL = "ws://localhost:8001/ws/transcribe_stream/"

# Audio file to stream
AUDIO_FILE = "/Users/a1197/Downloads/projet_machine_learning/audio_data/audioPLus698.wav"


async def send_heartbeat(websocket):
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(5)
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed")
            break


async def stream_audio_to_websocket(file_path, websocket_url):
    """
    Stream an audio file to the WebSocket server.
    """
    try:
        # Open a WebSocket connection
        all_stream = ""
        logger.info("launching websocket connection...")
        async with websockets.connect(websocket_url) as websocket:
            logger.info(f"Connected to WebSocket server: {websocket_url}")
            asyncio.create_task(send_heartbeat(websocket))
            # Open the audio file
            with sf.SoundFile(file_path, "rb") as audio_file:
                # Check file properties
                logger.info(
                    f"Streaming audio with sample rate: {audio_file.samplerate}, channels: {audio_file.channels}"
                )

                # Stream the file in chunks
                chunk_size = 16000  # Number of frames to send per message
                while True:
                    # Read the next chunk of frames
                    audio_chunk = audio_file.read(chunk_size, dtype="int16")  # Change dtype as needed
                    logger.info(f"audio_chunk.size, {audio_chunk.size}")
                    if not audio_chunk.size:
                        break  # End of file reached
                    # Send the audio chunk as binary data
                    await websocket.send(audio_chunk.tobytes())
                    # Optionally, receive a server response (if the server sends back responses)
                    response = await websocket.recv()

                    readable_text = arabic_reshaper.reshape(response)
                    logger.info(f"Server response: {get_display(readable_text)}")

                    all_stream += readable_text + " "
                logger.info("Sending end of stream signal...")
                await websocket.send(b"end_of_stream")  # End of stream signal as bytes
                logger.info("End of stream signal sent.")
            logger.info(f"Finished streaming audio: {get_display(all_stream)}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(stream_audio_to_websocket(AUDIO_FILE, WEBSOCKET_URL))
