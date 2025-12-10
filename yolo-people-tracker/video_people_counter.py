from ultralytics import YOLO
import cv2
import time
import csv
from pathlib import Path
import numpy as np

# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "yolov8s.pt"

VIDEO_PATH = r"C:\Users\mabdu\OneDrive\Desktop\yolo-people-tracker\data\hospital_corridor.mp4"

# AOI polygon (your wedge corridor zone)
# Order: around the polygon (clockwise or anti-clockwise)
AOI_POLY = np.array([
    [481, 134],   # top-left
    [576, 134],   # top-right
    [600, 468],   # bottom-right
    [187, 466],   # bottom-left
], dtype=np.int32)

# CSV log file path
LOG_PATH = Path("people_log.csv")

# Distance threshold (pixels) for deciding whether a center
# is "the same person" across frames when counting new entries
ENTRY_DISTANCE_THRESH = 50


def main():
    # Load YOLO model
    print(f"[INFO] Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Open video file
    print(f"[INFO] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] Could not open video file. Check path.")
        return

    # Prepare CSV (overwrite each run)
    with LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index",
            "timestamp_sec",
            "people_total",
            "people_in_zone",
            "entries_total",
        ])

        frame_index = 0
        prev_inside_centers: list[tuple[int, int]] = []
        total_entries = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video ended.")
                break

            frame_index += 1

            # Timestamp (seconds) from video
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Run YOLO inference on current frame
            start = time.time()
            results = model(frame)
            result = results[0]
            end = time.time()
            fps = 1.0 / (end - start) if (end - start) > 0 else 0.0

            people_total = 0
            people_in_zone = 0
            current_inside_centers: list[tuple[int, int]] = []

            # -----------------------------
            # Draw AOI polygon
            # -----------------------------
            cv2.polylines(
                frame,
                [AOI_POLY],
                isClosed=True,
                color=(255, 0, 0),   # BGR: blue-ish
                thickness=2,
            )

            # AOI label near first vertex
            label_pos = (int(AOI_POLY[0][0] + 10), int(AOI_POLY[0][1] - 10))
            cv2.putText(
                frame,
                "AOI",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            # Draw small circles on vertices (debug)
            for (px, py) in AOI_POLY:
                cv2.circle(frame, (int(px), int(py)), 5, (255, 0, 0), -1)

            # -----------------------------
            # Process YOLO detections
            # -----------------------------
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                if cls_name != "person":
                    continue

                people_total += 1

                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Center of bounding box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Check if center lies inside AOI polygon
                inside_zone = cv2.pointPolygonTest(AOI_POLY, (cx, cy), False) >= 0
                if inside_zone:
                    people_in_zone += 1
                    current_inside_centers.append((cx, cy))

                # Color: green inside AOI, cyan outside
                color = (0, 255, 0) if inside_zone else (0, 200, 200)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # Draw center point
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # -----------------------------
            # Entry counting (footfall)
            # -----------------------------
            new_entries = 0
            for (cx, cy) in current_inside_centers:
                is_new = True
                for (px, py) in prev_inside_centers:
                    dist_sq = (cx - px) ** 2 + (cy - py) ** 2
                    if dist_sq < ENTRY_DISTANCE_THRESH ** 2:
                        is_new = False
                        break
                if is_new:
                    new_entries += 1

            total_entries += new_entries
            prev_inside_centers = current_inside_centers

            # -----------------------------
            # HUD: counts and FPS
            # -----------------------------
            cv2.putText(
                frame,
                f"Total people: {people_total}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"In AOI: {people_in_zone}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Entries: {total_entries}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            # Show annotated frame
            cv2.imshow("Hospital Corridor - People Counter", frame)

            # Log to CSV
            writer.writerow([
                frame_index,
                f"{timestamp:.3f}",
                people_total,
                people_in_zone,
                total_entries,
            ])

            # Quit on 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Log saved to: {LOG_PATH.resolve()}")


if __name__ == "__main__":
    main()
