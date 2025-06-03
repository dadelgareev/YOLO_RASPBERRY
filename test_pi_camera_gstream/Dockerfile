FROM python:3.12-slim-bookworm

# Установка зависимостей для GStreamer, PyGObject, и инструментов сборки
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    v4l-utils \
    gcc \
    g++ \
    make \
    pkg-config \
    libcairo2-dev \
    python3-dev \
    ninja-build \
    meson \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY pi_camera_gstream.py .

CMD ["python", "pi_camera_gstream.py"]