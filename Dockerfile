FROM python:3.12-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt  && \
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

CMD ["python", "main.py"]