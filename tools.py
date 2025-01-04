import logging

import torch
import torchaudio
from lang_trans.arabic import buckwalter

logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
TARGET_AUDIO_SAMPLING = 16000
RESAMPLED_FILE = "resampled_audio_16kHz.wav"


def load_audio(file_path, save_resampled: bool = False):
    # Torchaudio loads the waveform and sampling rate
    speech_array, sampling_rate = torchaudio.load(file_path)
    # Resample to 16kHz (Wav2Vec2 models expect this sampling rate)
    if sampling_rate != TARGET_AUDIO_SAMPLING:
        logger.info(f"Resampling audio from {sampling_rate} Hz to 16000 Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=TARGET_AUDIO_SAMPLING)
        speech_array = resampler(speech_array)
        if save_resampled:
            torchaudio.save(RESAMPLED_FILE, resampler(speech_array).squeeze(), TARGET_AUDIO_SAMPLING)
    return speech_array.squeeze().numpy()


def transcribe_audio(file_path, processor, model, device):
    audio = load_audio(file_path)
    inputs = processor(audio, sampling_rate=TARGET_AUDIO_SAMPLING, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GP
    # Perform inference
    with torch.no_grad():
        logits = model(inputs.get("input_values")).logits
    # Get predicted tokens and decode to Arabic text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    # Return transcription in Arabic with Buckwalter encoding
    return buckwalter.untrans(transcription)


# Function to process multiple audio files
def transcribe_multiple_audio(files_list, processor, model, device):
    transcriptions = {}
    for file_path in files_list:
        if file_path.endswith(".wav"):
            transcription = transcribe_audio(f"./data/{file_path}", processor, model, device)
            transcriptions[file_path] = transcription
    return transcriptions
