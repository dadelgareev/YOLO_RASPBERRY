import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import signal
import sys
import logging
import os

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    filename='output/pi_camera.log',
    filemode='w',
    encoding='utf-8',
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

logger.info('Запущен тестовый модуль CSI-камеры с GStreamer')

running = True

def signal_handler(sig, frame):
    global running
    logger.info(f'Получен сигнал остановки (сигнал {sig})')
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    Gst.init(sys.argv)
    # Пайплайн GStreamer
    pipeline_str = (
        "v4l2src device=/dev/video2 ! "
        "video/x-raw,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! "
        "jpegenc ! "
        "avimux ! "
        "filesink location=output/output.avi"
    )
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except Exception as ex:
        logger.error(f"Ошибка создания пайплайна: {ex}")
        print(f"Ошибка: {ex}")
        sys.exit(1)

    os.makedirs('output', exist_ok=True)

    pipeline.set_state(Gst.State.PLAYING)
    logger.info("Начало записи видео")
    print("Начало записи видео")

    try:
        bus = pipeline.get_bus()
        frame_count = 0
        while running:
            msg = bus.timed_pop_filtered(1 * Gst.SECOND, Gst.MessageType.ANY)
            frame_count += 30
            if frame_count % 30 == 0:
                logger.info(f"Обработано кадров: {frame_count}")
                print(f"Обработано кадров: {frame_count}")
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    logger.error(f"Ошибка GStreamer: {err}, {debug}")
                    print(f"Ошибка GStreamer: {err}, {debug}")
                    break
                elif msg.type == Gst.MessageType.EOS:
                    logger.info("Конец потока")
                    print("Конец потока")
                    break
    except Exception as ex:
        logger.error(f"Ошибка: {ex}")
        print(f"Ошибка: {ex}")
    finally:
        pipeline.set_state(Gst.State.NULL)
        logger.info("Остановка записи")
        print("Остановка записи")

        output_file = 'output/output.avi'
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            logger.info(f"Размер выходного файла: {file_size} bytes")
            print(f"Размер выходного файла: {file_size} bytes")

if __name__ == '__main__':
    main()