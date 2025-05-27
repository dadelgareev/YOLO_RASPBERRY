FROM python:3.12-slim-bullseye

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["python", "main.py"]