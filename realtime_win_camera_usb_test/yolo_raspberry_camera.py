import pickle
import time
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort
import signal
import sys
import socket

HOST = '127.0.0.1'  # IP-адрес сервера
PORT = 65432        # Порт сервера

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='tracking.log', filemode='w', encoding='utf-8')

running = True

def signal_handler(sig, frame):
    global running
    logger.info("Получен сигнал остановки (Ctrl+C), завершаем работу...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')

# Инициализация трекера DeepSort
tracker = DeepSort(
    max_age=150,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.6,
    max_iou_distance=1.0,
)

# Захват видео с веб-камеры (0 - это индекс камеры по умолчанию)
cap = cv2.VideoCapture('run_human.mp4')

if not cap.isOpened():
    logger.error("Не удалось открыть веб-камеру")
    print("Ошибка: Не удалось открыть веб-камеру")
    exit(1)

logger.info('Обработка видеопотока началась.')

# Настройка параметров записи видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Расширение кадра: {frame_width}x{frame_height}")
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

tracked_objects_history = {}
frame_count = 0

while cap.isOpened() and running:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logger.info("Ошибка чтения кадра или конец потока")
        print("Ошибка: Не удалось прочитать кадр")
        break

    frame_count += 1

    # Обработка кадра с помощью YOLOv8
    results = model(frame, verbose=False, augment=True)
    detections = []

    if len(results) > 0 and results[0].boxes is not None:
        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = float(det.conf.cpu().numpy()[0])
            cls = int(det.cls.cpu().numpy()[0])

            # Фильтрация только людей (класс 0 в COCO) с высокой уверенностью
            if cls == 0 and conf > 0.7:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

    print(f"Список детекций: {detections}")
    # Обновление трекера
    tracks = tracker.update_tracks(detections, frame=frame)
    print(f"Список объектов треков: {tracks}")

    # Словарь объектов, зафиксированных за кадр
    current_frame_tracks = {}

    # Обработка треков
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        bbox = track.to_tlbr()  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox)

        # Проверяем валидность координат
        if x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
            # Сохраняем историю объекта
            if track_id not in tracked_objects_history:
                tracked_objects_history[track_id] = []
            tracked_objects_history[track_id].append({
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'frame_time': start_time,
                'frame_number': frame_count
            })

            # Сохраняем отслеживаемые объекты за кадр в словаре
            current_frame_tracks[track_id] = {
                'bbox': (x1, y1, x2, y2),
                'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                'frame_time': start_time,
                'frame_number': frame_count
            }
            # Рисуем bounding box и ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Передача словаря по TCP
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            serialized_data = pickle.dumps(current_frame_tracks)
            s.sendall(serialized_data)
            print("Данные отправлены:", current_frame_tracks)
    except Exception as e:
        print("Ошибка соединения:", e)

    # Запись кадра в выходное видео
    out.write(frame)

    # Отображение кадра в окне
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Получена команда остановки (q), завершаем работу...")
        running = False

    end_time = time.time()
    processing_time = end_time - start_time

    if frame_count % 60 == 0:
        logger.info(f'Кадр {frame_count} обработан за {processing_time:.4f} секунд')

cap.release()
out.release()
cv2.destroyAllWindows()
logger.info('Обработка видеопотока завершена.')

# Логирование статистики
logger.info("=== Статистика трекинга ===")
for track_id, history in tracked_objects_history.items():
    total_frames = len(history)
    first_frame = history[0]["frame_number"]
    last_frame = history[-1]["frame_number"]
    duration = last_frame - first_frame + 1

    logger.info(f'Track ID: {track_id}')
    logger.info(f'  - Кадров с объектом: {total_frames}')
    logger.info(f'  - Первое появление: кадр {first_frame}')
    logger.info(f'  - Последнее появление: кадр {last_frame}')
    logger.info(f'  - Продолжительность: {duration} кадров')
    logger.info(f'  - Последний BBox: {history[-1]["bbox"]}')
    logger.info(f'  - Последний центр: {history[-1]["center"]}')
    logger.info('---')

logger.info(f'Всего обработано кадров: {frame_count}')
logger.info(f'Всего уникальных объектов: {len(tracked_objects_history)}')