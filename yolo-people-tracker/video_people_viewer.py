from ultralytics import YOLO
import cv2
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_NAME = "yolov8s.pt"  # using small for speed on CPU
VIDEO_PATH = r"C:\Users\mabdu\OneDrive\Desktop\yolo-people-tracker\data\hospital_corridor.mp4"

def main():
    print(f"[INFO] Loading YOLO model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    print(f"[INFO] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("[ERROR] Could not open video file. Check path.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video ended.")
            break

        start = time.time()
        results = model(frame)
        result = results[0]
        end = time.time()
        fps = 1 / (end - start)

        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            # 💡 FILTER ONLY PERSON DETECTIONS
            if cls_name != "person":
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)
            conf = float(box.conf[0])

            # draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        cv2.imshow("Hospital Corridor - People Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
