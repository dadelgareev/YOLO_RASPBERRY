import time
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, filename='tracking.log', filemode='w', encoding='utf-8')

model = YOLO('yolov8n.pt')

tracker = DeepSort(
    max_age=150,
    n_init=3,
    nn_budget=100,
    max_cosine_distance=0.6,
    max_iou_distance=1.0,
)

gst_pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    logger.error("Не удалось открыть видеопоток с камеры")
    exit(1)

logger.info('Обработка видеопотока началась.')

frame_width, frame_height = 416, 416
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

tracked_objects_history = {}
frame_count = 0

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logger.info("Ошибка чтения кадра или конец потока")
        break

    frame_count += 1

    results = model(frame, verbose=False, augment=True)
    detections = []

    if len(results) > 0 and results[0].boxes is not None:
        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = float(det.conf.cpu().numpy()[0])
            cls = int(det.cls.cpu().numpy()[0])

            if cls == 0 and conf > 0.7:
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

    # Обновление трекера
    tracks = tracker.update_tracks(detections, frame=frame)

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

            # Рисуем bounding box и ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Вывод в консоль
            if frame_count % 60 == 0:
                logger.info(
                    f'Frame {frame_count} - Track ID: {track_id}, BBox: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})')

    out.write(frame)

    end_time = time.time()
    processing_time = end_time - start_time

    if frame_count % 60 == 0:
        logger.info(f'Кадр {frame_count} обработан за {processing_time:.4f} секунд')

cap.release()
out.release()
logger.info('Обработка видеопотока завершена.')

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