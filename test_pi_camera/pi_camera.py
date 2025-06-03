import logging
import signal
import sys
import cv2
import os
import time

logging.basicConfig(
    level=logging.INFO,
    filename='output/pi_camera.log',
    filemode='w',
    encoding='utf-8',
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

logger.info('Запущен тестовый модуль CSI-камеры')

running = True
frame_counter = 0

def signal_handler(sig, frame):
    global running
    logger.info(f'Получен сигнал остановки (сигнал {sig})')
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        logger.error("Не удалось открыть камеру на /dev/video0. Пробуем без V4L2...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Не удалось открыть камеру. Завершаем работу...")
            print("Ошибка: Не удалось открыть камеру")
            sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30


    logger.info(f"Камера открыта: {frame_width}x{frame_height}, FPS: {fps}")
    print(f"Камера открыта: {frame_width}x{frame_height}, FPS: {fps}")

    os.makedirs('output', exist_ok=True)

    output_file = 'output/output.avi'
    cap_out = cv2.VideoWriter(
        output_file,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (frame_width, frame_height)
    )

    if not cap_out.isOpened():
        logger.error("Не удалось создать выходной видеофайл")
        print("Ошибка: Не удалось создать выходной видеофайл")
        cap.release()
        sys.exit(1)

    logger.info("Начало записи видео")
    print("Начало записи видео")

    try:
        while running:
            ret, frame = cap.read()

            if not ret:
                logger.warning("Ошибка чтения кадра. Пропускаем...")
                print("Предупреждение: Ошибка чтения кадра")
                continue

            cap_out.write(frame)
            global frame_counter
            frame_counter += 1

            print(f"Считали и записали кадр {frame_counter}")

            if frame_counter % fps == 0:
                logger.info(f"Записан кадр: {frame_counter}")

            cv2.imwrite(f'output/frame-{frame_counter}.png', frame)


    except Exception as ex:
        logger.error(f"Ошибка во время записи: {ex}")
        print(f"Ошибка: {ex}")

    finally:
        logger.info("Остановка записи")
        print("Остановка записи")
        cap_out.release()
        cap.release()
        logger.info("Ресурсы освобождены")
        print("Ресурсы освобождены")
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"Размер выходного файла: {file_size} байт")
            print(f"Размер выходного файла: {file_size} байт")

if __name__ == '__main__':
    main()