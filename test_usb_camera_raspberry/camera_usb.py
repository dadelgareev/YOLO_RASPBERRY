import logging
import signal
import sys

import cv2

logging.basicConfig(level=logging.INFO, filename='camera_usb.log', filemode='w', encoding='utf-8')
logger = logging.getLogger(__name__)

logger.info('Запущен тестовый модуль камеры')

running = True # Статус камеры

def signal_handler(signal, frame):
    global running
    logger.info('Получен сигнал остановки камеры')
    running = False

# Регистрация сигналов (для остановки в докере через Ctrl + C)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    cap = cv2.VideoCapture(0) # 0 - камера по умолчанию

    if not cap:
        logger.error("Не удалось открыть камеру. Принудительное завершение работы...")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30

    cap_out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height),
    )
    logger.info("Начало записи")
    try:
        while running:
            ret, frame = cap.read()

            if not ret:
                logger.error("Ошибка чтения кадра. Приступаем к следующему")
                continue

            cap_out.write(frame)

            if cap.get(cv2.CAP_PROP_POS_FRAMES) % fps == 0:
                logger.info(f"Записан кадр: {cv2.CAP_PROP_POS_FRAMES}")
    except Exception as ex:
        logger.error(f"Ошибка во время записи: {ex}")
    finally:
        logger.info("Остановка записи.")
        cap.release()
        cap_out.release()
        logger.info("Ресурсы освобождены")

if __name__ == '__main__':
    main()
