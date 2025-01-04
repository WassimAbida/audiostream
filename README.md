
 # docker build -t websocket-app . 
# docker run -p 8000:8000 -v models:/models websocket-app 

uvicorn main:app --host 0.0.0.0 --port 8001 
run stream2