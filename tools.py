import torch
import torchaudio
from lang_trans.arabic import buckwalter


def load_audio(file_path):
    # Torchaudio loads the waveform and sampling rate
    speech_array, sampling_rate = torchaudio.load(file_path)
    # Resample to 16kHz (Wav2Vec2 models expect this sampling rate)
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def transcribe_audio(file_path, processor, model, device):
    audio = load_audio(file_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
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
