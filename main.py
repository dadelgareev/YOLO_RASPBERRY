import time
from norfair import Tracker
from norfair import Detection
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('run_human.mp4')
tracker = Tracker(distance_function="euclidean", distance_threshold=30)
logger.info('Video capture started.')
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for det in results.boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy()
        conf = det.conf.cpu().numpy()
        center_point = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Создаем Detection и сохраняем bbox координаты как атрибут
        detection = Detection(points=center_point, scores=conf)
        detection.bbox = (x1, y1, x2, y2)  # Сохраняем bbox
        detections.append(detection)

    tracked_objects = tracker.update(detections=detections)

    for obj in tracked_objects:
        if obj.last_detection is not None and hasattr(obj.last_detection, 'bbox'):
            x1, y1, x2, y2 = obj.last_detection.bbox
            track_id = obj.id

            # Рисуем bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Добавляем ID объекта
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f'Frame processed in {processing_time:.4f} seconds')
    cv2.imshow('Norfair Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
logger.info('Video capture ended.')
cv2.destroyAllWindows()