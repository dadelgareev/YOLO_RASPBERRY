import time
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model = YOLO('yolov8n.pt')

# Инициализация DeepSORT
tracker = DeepSort(
    max_age=300,
    n_init=1,
    nn_budget=300,
    max_cosine_distance = 0.7,
    max_iou_distance = 1
)

cap = cv2.VideoCapture('run_human.mp4')

if not cap.isOpened():
    logger.error("Не удалось открыть видео файл")
    exit(1)

logger.info('Video capture started.')

# Словарь для хранения истории объектов
tracked_objects_history = {}

frame_count = 0

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        logger.info("Достигнут конец видео или ошибка чтения кадра")
        break

    frame_count += 1

    results = model(frame, verbose=False)  # Отключаем verbose для уменьшения вывода
    detections = []

    if len(results) > 0 and results[0].boxes is not None:
        for det in results[0].boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
            conf = float(det.conf.cpu().numpy())
            cls = int(det.cls.cpu().numpy())

            # Фильтруем только людей (класс 0 в COCO dataset)
            if cls == 0 and conf > 0.7:  # Минимальная уверенность 0.7 (можно менять)
                # Формат для DeepSORT: [[x1, y1, w, h], conf, class]
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], conf, cls))

    # Обновление трекера
    tracks = tracker.update_tracks(detections, frame=frame)

    # Обработка треков (треки - отслеживаемые объекты)
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
                'frame_time': start_time,
                'frame_number': frame_count
            })

            # Рисуем bounding box и ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Вывод в консоль (только каждый 30-й кадр для уменьшения спама)
            if frame_count % 30 == 0:
                logger.info(
                    f'Frame {frame_count} - Track ID: {track_id}, BBox: (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})')

    end_time = time.time()
    processing_time = end_time - start_time

    # Выводим время обработки только каждый 30-й кадр
    if frame_count % 30 == 0:
        logger.info(f'Frame {frame_count} processed in {processing_time:.4f} seconds')

    cv2.save

    # Выход по нажатию 'q' или закрытию окна
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('DeepSORT Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
logger.info('Video capture ended.')

# Вывод статистики треков
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
    logger.info('---')

logger.info(f'Всего обработано кадров: {frame_count}')
logger.info(f'Всего уникальных объектов: {len(tracked_objects_history)}')