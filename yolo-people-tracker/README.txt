📍 Hospital Corridor People Tracking & Analytics System

A Computer Vision + Analytics pipeline that converts CCTV footage into real-time occupancy insights.

This project uses YOLOv8 + OpenCV to detect people inside a hospital corridor, applies a polygon AOI (Area of Interest) to monitor corridor movement only, logs temporal metrics, and visualizes insights on a Flask dashboard powered by Pandas & Chart.js.

The system acts as a prototype for hospital footfall monitoring, queue congestion analytics, and staff allocation optimization.

🔥 Key Features

🎯 Person detection using YOLOv8 (CPU-friendly setup)

📐 Polygon-based AOI for accurate corridor region filtering

🔢 Counts:

Total people in frame

People inside corridor zone

Entry events (unique crossings)

🧠 Centroid-based temporal event logic for footfall

⏱ Real-time FPS monitoring per frame

📁 CSV logging with timestamps for further analytics/BI

📊 Flask dashboard with interactive Chart.js trends

📎 Fully modular code & easy to extend with trackers/sensors

🧱 Tech Stack
Component	Tech
Detection	YOLOv8 (Ultralytics)
Processing	Python + OpenCV
Tracking Logic	Custom centroid distance matcher
Data Logging	CSV via pandas
Visualization	Flask + Chart.js
Future Scaling	DeepSORT / ByteTrack + RTSP + BI
📂 Project Structure
yolo-people-tracker/
│── video_people_counter.py        # Main detection & logging pipeline
│── analytics_server.py            # Flask dashboard server
│── people_log.csv                 # Logged analytics output
│── templates/
│   └── dashboard.html             # Visualization interface
│── data/
│   └── hospital_corridor.mp4      # Input video source
│── models/
│   └── yolov8s.pt                 # Model weights (optional)

🚀 Run Instructions
1. Install dependencies
pip install ultralytics opencv-python flask pandas

2. Run people detection + logging
python video_people_counter.py


This generates people_log.csv.

3. Launch dashboard
python analytics_server.py


Visit:

http://127.0.0.1:5000/

📊 Output & Insights

Metrics generated:

people_total → total visible detections per frame

people_in_zone → detections within corridor AOI

entries_total → cumulative unique people entering zone

timestamp_sec → time-based analytics support

Sample visualization:

✔ occupancy evolution over time
✔ peak load periods
✔ total corridor usage count

Perfect for operations monitoring & crowd behavior understanding.

🧭 Future Improvements

DeepSORT/ByteTrack ID tracking for consistent re-identification

Live RTSP CCTV feed support

Real-time anomaly alerting (overcrowding/queue buildup)

PowerBI streaming integration

Heatmaps for movement density

📌 Why This Project Matters

Hospitals often struggle to measure movement load and peak times.
This system turns raw CCTV into quantifiable metrics, enabling:

Better staff allocation

Patient flow management

Reduced congestion & wait times

Data-backed decisions

💬 Contact / Contribution

Contributions, discussions & improvements are welcome.
If you'd like to collaborate or scale this to live deployment — reach out!