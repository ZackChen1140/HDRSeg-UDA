FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN pip3 install torch torchvision torchaudio

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt