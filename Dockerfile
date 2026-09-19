FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN mkdir -p /app/data/input /app/data/output /app/models /app/src

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"
RUN python3 -c "import mediapipe"

COPY weights/ /app/models/
COPY src/ /app/src/
COPY main.py .
COPY README.md .

CMD ["python3", "main.py"]