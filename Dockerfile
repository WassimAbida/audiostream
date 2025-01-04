# Use official python runtime as a base image
FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libsndfile1 libgomp1 && \
    rm -rf /var/lib/apt/lists/*
# Set working directory to /app
WORKDIR /app

# Copy local requirements file to container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from the local directory to the container
COPY main.py .
COPY tools.py .
# COPY /models /models 

# Expose port for FastAPI
EXPOSE 8000

# Command to run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
