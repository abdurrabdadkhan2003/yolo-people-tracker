# YOLO People Tracking System

> Real-world CCTV analytics pipeline using YOLOv8 for integrated people detection, tracking, and area-of-interest (AOI) zone analytics.

**Status**: Production-ready | **Last Updated**: December 2025

---

## Project Overview

This is a complete end-to-end computer vision system designed for real-world CCTV surveillance applications. It integrates:
- YOLOv8 object detection for real-time people detection
- Polygon-based AOI zone counting for spatial analytics
- CSV logging for data persistence and historical analysis
- Flask dashboard for live monitoring and visualization

Ideal for hospital corridors, retail spaces, or any scenario requiring real-time crowd monitoring.

---

## Key Features

- Real-time Detection: YOLOv8 inference for high-speed people detection
- Spatial Analytics: Polygon AOI zone counting with geometry-based logic
- Data Logging: CSV export for downstream analysis and compliance
- Web Dashboard: Flask-based live monitoring interface
- Modular Architecture: Easily configurable for different scenarios
- Production-Ready: Error handling, logging, and performance optimization

---

## Technical Stack

- Detection: YOLOv8 (Ultralytics)
- Computer Vision: OpenCV  
- Geometry: Shapely (polygon operations)
- Backend: Flask
- Data: Pandas, CSV
- Frontend: HTML5, JavaScript, Plotly

---

## Real-World Use Cases

1. Hospital Corridors: Monitor patient flow and staff movement
2. Retail Spaces: Track customer density and hotspots
3. Airports/Stations: Real-time crowd monitoring
4. Office Buildings: Occupancy tracking for energy management
5. Events: Crowd safety monitoring

---

## License

MIT License - Feel free to use this project for educational and commercial purposes.

Built with focus on real-world computer vision applications
