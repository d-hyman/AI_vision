import cv2
from ultralytics import YOLO

CAMERA_INDEX = 0
MODEL_NAME = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.5

#main function
def main():
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(CAMERA_INDEX)    #opens video stream

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        height, width = frame.shape[:2]
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]

            # relative position: 0,0 = frame center, -1/1 = left-top/right-bottom edges
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rel_x = (cx - width / 2) / (width / 2)
            rel_y = (cy - height / 2) / (height / 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f} ({rel_x:.2f}, {rel_y:.2f})"
            cv2.putText(frame, label, (int(x1), max(int(y1) - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.drawMarker(frame, (width // 2, height // 2), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1)

        cv2.imshow("AI Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

#if file run in terminal is main.py, run main function
if __name__ == "__main__":
    main()
