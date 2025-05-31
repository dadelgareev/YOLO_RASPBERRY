FROM python:3.12-slim-bullseye

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY yolo_raspberry_camera.py .

# Команда для запуска
CMD ["python", "yolo_raspberry_camera.py"]

