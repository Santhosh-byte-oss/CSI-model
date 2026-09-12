Wi-Fi CSI Edge AI Spatial Monitoring System

An end-to-end, privacy-preserving edge-IoT monitoring system that detects human presence and motion states (standing vs. walking) using standard Wi-Fi Channel State Information (CSI), deployed on bare-metal microcontrollers and visualized through an immersive Three.js 3D cinematic web dashboard.

Architecture & System Pipeline

Plaintext

[ESP32: CSI Sensing & TinyML Inference] --(Serial/UART)--> [Python Flask-SocketIO Server]
[ESP8266: RSSI Triangulation Node]       --(UDP Broadcast)--> [Socket.io Real-Time Stream] ---> [Three.js 3D Dashboard]
ESP32 (Primary Node): Captures raw physical Wi-Fi CSI amplitudes, performs real-time digital signal processing (DSP) feature extraction (mean, max, and variance differences), and runs hardware-level TinyML inference using a compiled Random Forest model.

ESP8266 (Secondary Node): Tracks RSSI spatial telemetry and broadcasts positional packets over UDP.

Python Backend:
An asynchronous Flask-SocketIO bridge that aggregates serial streams and UDP packets, applies rolling-window smoothing, and broadcasts real-time telemetry.

Frontend Dashboard:
A high-tech web interface built with Three.js featuring volumetric lighting, a dynamic grid, and a real-time GLTF animated avatar (avatar.glb) that mirrors user positioning and skeletal animations.

Project Structure

Plaintext
CSI_Project/
│
├── ESP32_Edge_AI/             # Arduino firmware for CSI capture & TinyML
├── ESP8266_RSSI_Node/         # Wi-Fi RSSI telemetry node
├── static/
│   ├── avatar.glb             # 3D character asset for Three.js
│   └── ...
├── templates/
│   └── index.html             # Three.js web dashboard frontend
├── 10_web_dashboard.py        # Flask-SocketIO central server & loop
├── CSI_Model.h                # Exported TinyML decision tree header
└── requirements.txt           # Python dependencies
Prerequisites & Dependencies

Hardware Requirements

ESP32 Node (configured for CSI extraction)

ESP8266 Node (for RSSI spatial tracking)

Micro-USB data cables

Software & Libraries
Python 3.8+ with the following packages installed:

Bash
pip install flask flask-socketio pyserial numpy scikit-learn
Arduino IDE with libraries:

micromlgen (for exporting scikit-learn models to C++)

ESP32 Wi-Fi and ESPAsyncWebServer libraries

Getting Started & Execution
Flash the Microcontrollers:

Open ESP32_Edge_AI/ESP32_Edge_AI.ino in the Arduino IDE, update your Wi-Fi credentials, and upload to the ESP32.

Flash the secondary tracking script onto the ESP8266.

Start the Python Backend:
Navigate to the project directory and launch the centralized server:

Bash
python 10_web_dashboard.py
Open the Web Dashboard:
Open your browser and navigate to:

Plaintext
http://127.0.0.1:5000
License
This project is open-source and available under the MIT License.

## Dashboard Preview

![Three.js CSI Dashboard Preview](assets/dashboard.png)