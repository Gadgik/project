import requests
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import gradio as gr
from mpl_toolkits.mplot3d import Axes3D
import psycopg2

# Подключение к базе данных PostgreSQL
DB_URL = "postgresql://postgres:1234@localhost:5432/postgres"

# Глобальная переменная для API URL
API_URL = "https://olimp.miet.ru/ppo_it/api"

# Подключение к базе данных и создание таблиц при необходимости
def init_db():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # Таблица для хранения карты
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mars_map (
            map_data BYTEA
        )
    """)
    
    # Таблица для хранения координат модулей и стоимости установки базовых станций
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS module_data (
            id SERIAL PRIMARY KEY,
            sender_x INT,
            sender_y INT,
            listener_x INT,
            listener_y INT,
            cuper_price FLOAT,
            engel_price FLOAT
        )
    """)
    
    # Таблица для хранения информации о базовых станциях
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            id SERIAL PRIMARY KEY,
            x INT,
            y INT,
            type VARCHAR(10)
        )
    """)
    
    conn.commit()
    cursor.close()
    conn.close()

# Функция для запроса тайла
def fetch_tile():
    response = requests.get(f"{API_URL}")
    if response.status_code == 200:
        return np.array(response.json()["message"]["data"], dtype=np.uint8)
    return None

# Функция для получения данных о модулях и стоимости
def fetch_coords_and_prices():
    response = requests.get(f"{API_URL}/coords")
    if response.status_code == 200:
        data = response.json()["message"]
        return data["sender"], data["listener"], data["price"]
    return (None, None, None)

# Функция для сборки карты из 16 тайлов
def assemble_map():
    full_map = np.zeros((256, 256), dtype=np.uint8)
    for row in range(4):
        for col in range(4):
            while True:
                tile = fetch_tile()
                if tile is not None and check(tile, full_map):
                    full_map[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] = tile
                    break
    return full_map

def check(tile, full_map):
    for row in range(4):
        for col in range(4):
            if sum(sum(full_map[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64] == tile)) == 64 * 64:
                return False
    return True

# Функция для поиска возвышенностей (локальных максимумов)
def find_peaks(height_map):
    peaks = []
    for x in range(1, 255):
        for y in range(1, 255):
            if height_map[x, y] == np.max(height_map[x-1:x+2, y-1:y+2]):
                peaks.append((x, y, height_map[x, y]))
    return peaks

# Сохранение карты в базу данных
def save_map_to_db(map_data):
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # Удаляем предыдущие данные карты
    cursor.execute("DELETE FROM mars_map")
    
    # Сохраняем новую карту
    cursor.execute("INSERT INTO mars_map (map_data) VALUES (%s)", (map_data.tobytes(),))
    
    conn.commit()
    cursor.close()
    conn.close()

# Сохранение данных о модулях и стоимости в базу данных
def save_module_data_to_db(sender, listener, cuper_price, engel_price):
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # Удаляем предыдущие данные
    cursor.execute("DELETE FROM module_data")
    
    # Вставляем новые данные
    cursor.execute(
        "INSERT INTO module_data (sender_x, sender_y, listener_x, listener_y, cuper_price, engel_price) VALUES (%s, %s, %s, %s, %s, %s)",
        (sender[0], sender[1], listener[0], listener[1], cuper_price, engel_price)
    )
    
    conn.commit()
    cursor.close()
    conn.close()

# Сохранение данных о станциях в базу данных
def save_stations_to_db(stations):
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    
    # Удаляем предыдущие данные
    cursor.execute("DELETE FROM stations")
    
    # Вставляем новые данные
    for x, y, station_type in stations:
        cursor.execute(
            "INSERT INTO stations (x, y, type) VALUES (%s, %s, %s)",
            (x, y, station_type)
        )
    
    conn.commit()
    cursor.close()
    conn.close()

# Функция для 3D визуализации карты
def plot_3d_map(angle=30):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(256), np.arange(256))
    ax.plot_surface(X, Y, mars_map, cmap='viridis')
    ax.view_init(elev=30, azim=angle)
    ax.set_title("3D карта Марса")
    return fig

# Функция для вращения карты в реальном времени
def rotate_map(angle):
    return plot_3d_map(angle)

# Подготовка данных и сохранение в базу
init_db()

# Получаем координаты модулей и стоимость станций
sender, listener, prices = fetch_coords_and_prices()
cuper_price, engel_price = prices
save_module_data_to_db(sender, listener, cuper_price, engel_price)

# Собираем карту
mars_map = assemble_map()
save_map_to_db(mars_map)

# Ищем пики для установки базовых станций
peaks = find_peaks(mars_map)
stations = []
for x, y, height in peaks:
    if height > 200:  # Условие для выбора высоты
        if len(stations) % 2 == 0:
            stations.append((x, y, "Cuper"))
        else:
            stations.append((x, y, "Engel"))
save_stations_to_db(stations)

# Интерфейс Gradio для 3D карты
demo = gr.Interface(
    fn=rotate_map,
    inputs=gr.Slider(0, 360, step=5, label="Угол поворота"),
    outputs=gr.Plot(),
    title="Марсианская карта",
    description="Трехмерная карта Марса с возможностью вращения"
)

demo.launch()
