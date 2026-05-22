FROM python:3.10-slim

# Install system dependencies for audio and networking
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY . .

# Default port for gRPC
EXPOSE 50051

ENV PYTHONPATH=/app/app

CMD ["python", "app/node.py"]