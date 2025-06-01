import socket
import pickle

HOST = '0.0.0.0' # слушаем все интерфейсы
PORT = 65432

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    print('Модуль процессинга начинает прослушку')

    while True:
        conn, addr = s.accept()
        with conn:
            print("Подключение от", addr)
            data = b''
            while True:
                part = conn.recv(1024)
                if not part:
                    break
                data += part
            try:
                received_dict = pickle.loads(data)
                print("Получен словарь:", received_dict)
            except Exception as e:
                print("Ошибка при десериализации:", e)