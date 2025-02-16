import os

from locust import HttpUser, between, task


class TranscribeUser(HttpUser):
    wait_time = between(1, 3)  # Simulated wait time between requests

    @task
    def upload_audio_file(self):
        file_path = "data/audioPLus211.wav"  # Replace with your actual test audio file

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Test file {file_path} not found!")
            return

        # Open the file in binary mode for upload
        with open(file_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}  # Adjust MIME type if needed
            response = self.client.post("/transcribe/", files=files)

        # Print the response for debugging
        print(response)
