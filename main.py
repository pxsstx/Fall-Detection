# นำเข้าไลบรารีที่จำเป็น
import cv2  # ไลบรารี OpenCV สำหรับงานคอมพิวเตอร์วิชั่น
from ultralytics import YOLO  # โมเดล YOLOv8 สำหรับการตรวจจับวัตถุ
import cvzone  # ไลบรารี OpenCV สำหรับการแสดงข้อความที่ง่ายขึ้น

# สร้างหน้าต่างชื่อ 'Fall Detection'
cv2.namedWindow('Fall Detection')

# โหลดโมเดล YOLO ที่ผ่านการฝึกมาแล้ว (ตรวจจับคน)
model = YOLO("best.pt")  # เปลี่ยนชื่อไฟล์เป็นโมเดลของคุณ
names = model.model.names  # ดึงชื่อของคลาสจากโมเดล เช่น 'person'

# เปิดไฟล์วิดีโอ (หรือใช้กล้องโดยเปลี่ยนเป็น index 0)
cap = cv2.VideoCapture('fall.mp4')  # หรือใช้ 0 สำหรับกล้อง

count = 0  # ตัวนับเฟรม

# วนลูปเพื่อประมวลผลเฟรมจากวิดีโอ
while True:
    ret, frame = cap.read()  # อ่านเฟรม
    if not ret:
        break  # ออกจากลูปเมื่อวิดีโอจบ

    count += 1
    if count % 3 != 0:
        continue  # ประมวลผลแค่ทุก 3 เฟรม เพื่อความเร็ว

    frame = cv2.resize(frame, (1020, 600))  # ปรับขนาดเฟรมให้เหมาะสม

    # ตรวจจับและติดตามคน
    results = model.track(frame, persist=True)

    # ตรวจสอบว่ามีการตรวจจับและมี track id หรือไม่
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # ดึงข้อมูลจากผลลัพธ์
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # วนลูปแต่ละวัตถุที่ตรวจจับได้
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box
            h = y2 - y1  # ความสูงของบ็อกซ์
            w = x2 - x1  # ความกว้างของบ็อกซ์
            thresh = h - w  # ความต่างระหว่างความสูงกับความกว้าง

            print(f"TrackID {track_id} | Height: {h}, Width: {w} → Thresh: {thresh}")

            if thresh <= 0:
                # แสดงผลคนที่ล้มด้วยกรอบสีแดง
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cvzone.putTextRect(frame, f'Fall ({track_id})', (x1, y1 - 10), 1, 1, (0, 0, 255), 2)
            else:
                # แสดงผลคนที่ปกติด้วยกรอบสีเขียว
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'Person ({track_id})', (x1, y1 - 10), 1, 1, (0, 255, 0), 2)

    # แสดงผลลัพธ์ในหน้าต่าง 'Fall Detection'
    cv2.imshow("Fall Detection", frame)

    # กด 'q' เพื่อออกจากลูป
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ปิดการจับภาพและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()
