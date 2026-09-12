import warnings
warnings.filterwarnings("ignore")

import socket
import serial
import threading
import time
import numpy as np
from collections import deque, Counter
from flask import Flask, render_template
from flask_socketio import SocketIO

# --- CONFIGURATION ---
ESP32_PORT = 'COM3'
BAUD_RATE = 921600

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

labels = {0: 'NO HUMAN', 1: 'STANDING', 2: 'WALKING'}
latest_csi_line = None
latest_rssi_val = "N/A"
data_lock = threading.Lock()

def read_esp32_thread():
    global latest_csi_line
    print(f"Starting ESP32 thread on {ESP32_PORT}...")
    while True:
        try:
            ser_32 = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=0.1)
            print("ESP32 Connected Successfully!")
            while True:
                line = ser_32.readline().decode('utf-8', errors='ignore').strip()
                
                # THE FIX: Python now listens for the new Edge AI keywords!
                if line.startswith('SYS:') or line.startswith('AI_'):
                    with data_lock:
                        latest_csi_line = line
                elif not line:
                    time.sleep(0.001)
                    
        except Exception as e:
            print(f"ESP32 Disconnected! Retrying... ({e})")
            time.sleep(2)

def read_esp8266_thread():
    global latest_rssi_val
    print("Starting ESP8266 Wi-Fi listener on UDP port 4210...")
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(('0.0.0.0', 4210))
    udp_socket.settimeout(0.5)
    while True:
        try:
            data, addr = udp_socket.recvfrom(1024)
            line = data.decode('utf-8').strip()
            if line.startswith('NODE2'):
                parts = line.split(',')
                if len(parts) >= 2:
                    with data_lock:
                        latest_rssi_val = parts[1]
        except socket.timeout:
            pass
        except Exception as e:
            time.sleep(1)

def processing_and_prediction_loop():
    global latest_csi_line, latest_rssi_val
    rssi_buffer = deque(maxlen=15)
    prediction_buffer = deque(maxlen=10) 

    print("Entering Edge AI monitoring loop...")
    while True:
        with data_lock:
            line = latest_csi_line
            rssi = latest_rssi_val
            latest_csi_line = None

        if not line:
            socketio.sleep(0.005)
            continue

        try:
            if line.startswith('SYS:CALIBRATION_COMPLETE'):
                print("\nESP32 hardware calibration complete! Dual-node active.")
                socketio.emit('csi_data', {'mode': 'calibrating', 'progress': 100})
                socketio.sleep(1)
                continue

            if line.startswith('AI_PREDICT:'):
                parts = line.split(',')
                raw_prediction = int(parts[0].split(':')[1])
                var_diff = float(parts[1].split(':')[1])
                
                prediction_buffer.append(raw_prediction)
                most_common_pred = Counter(prediction_buffer).most_common(1)[0][0]
                status = labels.get(most_common_pred, str(most_common_pred).capitalize())
                
                motion_intensity = min(int((var_diff / 80.0) * 100), 100)
                
                try:
                    rssi_num = float(rssi)
                    rssi_buffer.append(rssi_num)
                    smoothed_rssi = np.mean(rssi_buffer)
                    empty_room_rssi = -50.0 
                    estimated_x = (smoothed_rssi - empty_room_rssi) * 0.4
                    estimated_x = max(min(estimated_x, 5.0), -5.0)
                except:
                    estimated_x = 0.0

                display_rssi = str(int(np.mean(rssi_buffer))) if len(rssi_buffer) > 0 else "N/A"

                socketio.emit('csi_data', {
                    'mode': 'live',
                    'status': status,
                    'intensity': motion_intensity,
                    'x_pos': estimated_x,
                    'node2_rssi': display_rssi
                })
        except Exception:
            pass 
        socketio.sleep(0.005)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    threading.Thread(target=read_esp32_thread, daemon=True).start()
    threading.Thread(target=read_esp8266_thread, daemon=True).start()
    print("Starting Web Server at http://127.0.0.1:5000")
    socketio.start_background_task(processing_and_prediction_loop)
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)