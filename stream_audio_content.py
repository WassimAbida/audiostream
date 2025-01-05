import asyncio
import logging

import arabic_reshaper
import librosa
import soundfile as sf
import websockets
from bidi.algorithm import get_display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

WEBSOCKET_URL = "ws://fastapi:8000/ws/transcribe_stream/"
AUDIO_FILE = "data/audioPLus211.wav"
TARGET_AUDIO_SAMPLING = 16000
CHUNK_SIZE = 16000  # Number of frames to send per message (1 second of audio)


# Heartbeat mechanism to maintain WebSocket connection
async def send_heartbeat(websocket):
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(5)
        except websockets.exceptions.ConnectionClosed:
            logger.error("Connection closed")
            break


# Resample audio if the sample rate differs from the target rate
def resample_audio(audio_chunk, original_rate, target_rate):
    logger.info(f"Resampling audio from {original_rate}Hz to {target_rate}Hz")
    audio_float = audio_chunk.astype("float32") / 32768.0  # Normalize to [-1, 1]
    resampled_audio = librosa.resample(audio_float, orig_sr=original_rate, target_sr=target_rate)
    logger.info(f"Resampled audio, {resampled_audio.shape}")
    return (resampled_audio * 32768.0).astype("int16")


async def stream_audio_to_websocket(file_path, websocket_url):
    """
    Stream an audio file to the WebSocket server.
    """
    try:
        # Open a WebSocket connection
        logger.info("Launching WebSocket connection...")
        async with websockets.connect(websocket_url) as websocket:
            logger.info(f"Connected to WebSocket server: {websocket_url}")
            asyncio.create_task(send_heartbeat(websocket))
            all_stream = ""

            # Open the audio file
            with sf.SoundFile(file_path, "rb") as audio_file:
                logger.info(
                    f"Streaming audio with sample rate: {audio_file.samplerate}, channels: {audio_file.channels}"
                )

                # Ensure the sample rate is consistent before streaming
                if audio_file.samplerate != TARGET_AUDIO_SAMPLING:
                    full_audio = audio_file.read(dtype="int16")
                    full_audio_resampled = resample_audio(full_audio, audio_file.samplerate, TARGET_AUDIO_SAMPLING)
                    audio_file = sf.SoundFile(full_audio_resampled, mode="r", samplerate=TARGET_AUDIO_SAMPLING)

                # Stream the file in chunks

                while True:
                    audio_chunk = audio_file.read(CHUNK_SIZE, dtype="int16")
                    if not audio_chunk.size:
                        break  # End of file reached

                    # Send the audio chunk as binary data
                    await websocket.send(audio_chunk.tobytes())
                    try:
                        # Set a timeout for receiving a response from the server
                        response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    except asyncio.TimeoutError:
                        logger.warning("No response from server in time. Breaking the loop.")
                        break  # Exit the loop if server response times out

                    # Reshape Arabic text for better display
                    readable_text = arabic_reshaper.reshape(response)
                    logger.info(f"Server response: {get_display(readable_text)}")

                    all_stream += readable_text + " "

                # Send the end of stream signal
                logger.info("Sending end of stream signal...")
                await websocket.send(b"end_of_stream")
                logger.info("End of stream signal sent.")

            logger.info(f"Finished streaming audio: {get_display(all_stream)}")
    except Exception as e:
        logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(stream_audio_to_websocket(AUDIO_FILE, WEBSOCKET_URL))
