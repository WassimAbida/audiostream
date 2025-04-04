
# Web app for streamed audio transcription

Transcript Arabic streamed audio into text using FastAPI, Speech2Text AI models & docker


## (Recommended) Launch using Docker

first start by launching the inference server built with fastapi,
 downloading model might take some time for first launch

```bash
docker-compose up fastapi
```

service is accessible on [http://localhost:8000/docs](http://0.0.0.0:8000/docs)

Next launch the streaming server, the client sending audio data for transcription based on wav2vec model from huggingFace [jonatasgrosman/wav2vec2-large-xlsr-53-arabic](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-arabic)

streaming server is only for demo purpose, in real world application, this process will be substituted by a microphone or audio emitter sending data to the inference server

```bash
docker-compose up streaming
```

### Project Setup - Manual

```bash
peorty install
poetry run uvicorn main:app --host 0.0.0.0 --port 8001
poetry run python -m stream_audio_content
```

### Launch Load test using locust server

```bash
locust -f locustfile.py --host=http://localhost:8000
```

