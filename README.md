
## Web app for streamed audio transcription

Transcript Arabic streamed audio into text using FastAPI, Speech2Text AI models & docker

### Project Setup - Manual

```bash
peorty install

poetry run uvicorn main:app --host 0.0.0.0 --port 8001
poetry run python -m stream_audio_content

```
### Launch using Docker
```bash
docker-compose up fastapi
docker-compose up streaming
```

### Launch Load test using locust server
locust -f locustfile.py --host=http://localhost:8000
