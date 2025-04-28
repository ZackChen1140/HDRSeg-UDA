FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -ms /bin/bash user0

USER user0

WORKDIR /home/user0/app

RUN pip3 install torch torchvision torchaudio

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt