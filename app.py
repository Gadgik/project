import requests
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from mpl_toolkits.mplot3d import Axes3D
import psycopg2

# Подключение к базе данных PostgreSQL
DB_URL = "postgresql://postgres:1234@localhost:5432/postgres"

# Глобальная переменная для API URL
API_URL = "https://olimp.miet.ru/ppo_it/api"

def set_api_url(api_url):
    global API_URL
    API_URL = api_url

def init_db():
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mars_map (
            map_data BYTEA
        )
    """)
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

def fetch_tile():
    response = requests.get(f"{API_URL}")
    if response.status_code == 200:
        return np.array(response.json()["message"]["data"], dtype=np.uint8)
    return None

def fetch_coords_and_prices():
    response = requests.get(f"{API_URL}/coords")
    if response.status_code == 200:
        data = response.json()["message"]
        return data["sender"], data["listener"], data["price"]
    return (None, None, None)

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
            if np.array_equal(full_map[row * 64:(row + 1) * 64, col * 64:(col + 1) * 64], tile):
                return False
    return True

def find_peaks(height_map):
    peaks = []
    for x in range(1, 255):
        for y in range(1, 255):
            if height_map[x, y] == np.max(height_map[x-1:x+2, y-1:y+2]):
                peaks.append((x, y, height_map[x, y]))
    return peaks

def plot_3d_map(angle=30):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(256), np.arange(256))
    ax.plot_surface(X, Y, mars_map, cmap='viridis')
    ax.view_init(elev=30, azim=angle)
    ax.set_title("3D карта Марса")
    return fig

def visualize_modules():
    fig, ax = plt.subplots()
    ax.imshow(mars_map, cmap='gray')
    ax.scatter([sender[0], listener[0]], [sender[1], listener[1]], c='red', label='Модули')
    ax.legend()
    return fig

def visualize_stations():
    fig, ax = plt.subplots()
    ax.imshow(mars_map, cmap='gray')
    for x, y, station_type in stations:
        color = 'blue' if station_type == "Cuper" else 'green'
        ax.scatter(x, y, c=color, label=station_type)
    ax.legend()
    return fig

def visualize_coverage():
    fig, ax = plt.subplots()
    ax.imshow(mars_map, cmap='gray')
    for x, y, station_type in stations:
        radius = 32 if station_type == "Cuper" else 64
        circle = plt.Circle((x, y), radius, color='blue' if station_type == "Cuper" else 'green', fill=False)
        ax.add_patch(circle)
    return fig

def station_count():
    cuper_count = sum(1 for _, _, t in stations if t == "Cuper")
    engel_count = sum(1 for _, _, t in stations if t == "Engel")
    total_cost = cuper_count * cuper_price + engel_count * engel_price
    return f"Cuper: {cuper_count}, Engel: {engel_count}, Total Cost: {total_cost:.2f} байткоинов"

init_db()
sender, listener, prices = fetch_coords_and_prices()
cuper_price, engel_price = prices
mars_map = assemble_map()
peaks = find_peaks(mars_map)
stations = [(x, y, "Cuper" if i % 2 == 0 else "Engel") for i, (x, y, h) in enumerate(peaks) if h > 200]

demo = gr.Interface(
    fn=station_count,
    inputs=[],
    outputs="text",
    title="Марсианская карта",
    description="Анализ базовых станций"
)

map_demo = gr.Interface(
    fn=plot_3d_map,
    inputs=gr.Slider(0, 360, step=5, label="Угол поворота"),
    outputs=gr.Plot()
)

modules_demo = gr.Interface(
    fn=visualize_modules,
    inputs=[],
    outputs=gr.Plot()
)

stations_demo = gr.Interface(
    fn=visualize_stations,
    inputs=[],
    outputs=gr.Plot()
)

coverage_demo = gr.Interface(
    fn=visualize_coverage,
    inputs=[],
    outputs=gr.Plot()
)

api_input = gr.Interface(
    fn=set_api_url,
    inputs=gr.Textbox(label="API URL"),
    outputs=[]
)

gr.TabbedInterface(
    [api_input, map_demo, modules_demo, stations_demo, coverage_demo, demo],
    ["API", "Карта", "Модули", "Станции", "Зоны покрытия", "Статистика"]
).launch()