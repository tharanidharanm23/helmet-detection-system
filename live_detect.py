import cv2
from ultralytics import YOLO
from collections import defaultdict, deque

# load trained model
model = YOLO("best.pt")

# open webcam
cap = cv2.VideoCapture(0)

# store history per tracked ID
history = defaultdict(lambda: deque(maxlen=7))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO + tracking (THIS IS THE KEY)
    results = model.track(
        frame,
        conf=0.3,
        iou=0.5,
        persist=True,   # keeps ID across frames
        tracker="bytetrack.yaml"
    )

    if results and results[0].boxes is not None:
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else None

            if conf < 0.4 or track_id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # store decision history for this person
            history[track_id].append(cls)

            helmet_votes = history[track_id].count(0)
            no_helmet_votes = history[track_id].count(1)

            if helmet_votes >= no_helmet_votes:
                label = f"WITH HELMET (ID {track_id})"
                color = (0, 255, 0)
            else:
                label = f"NO HELMET (ID {track_id})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("Helmet Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
