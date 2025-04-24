import cv2
from ultralytics import YOLO

# โหลดโมเดลที่เทรนแล้ว
model = YOLO("best.pt")

# เปิดวิดีโอ
cap = cv2.VideoCapture("fall.mp4")

while True:
    success, frame = cap.read()
    if not success:
        break

    # ใช้โมเดลตรวจจับในแต่ละเฟรม
    results = model(frame, stream=True)

    # วาดกล่องผลลัพธ์ลงบนเฟรม
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # ตำแหน่งกรอบ
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = model.names[cls]

            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงผล
    cv2.imshow("YOLOv11 Fall Detection", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้อง/วิดีโอ และหน้าต่าง
cap.release()
cv2.destroyAllWindows()
