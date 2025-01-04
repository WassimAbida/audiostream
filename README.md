
## Web app for streamed audio transcription

Transcript Arabic streamed audio into text using FastAPI, Speech2Text AI models & docker
```bash
docker build -t websocket-app .
docker run -p 8000:8000 -v models:/models websocket-app
```

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
python -m stream_audio_content
```


```bash
docker-compose up fastapi
docker-compose up streaming
```
