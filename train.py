from ultralytics import YOLO

# โหลดโมเดล YOLOv11
model = YOLO("yolo11s.pt")

# ฝึกโมเดลด้วยข้อมูลที่เตรียมไว้
model.train(
    data="datasets/data.yaml",    # ไฟล์กำหนดเส้นทางของชุดข้อมูลและคลาส
    imgsz=640,           # ขนาดภาพ
    epochs=50,           # จำนวนรอบการฝึก
    batch=16,            # ขนาดแบทช์
    device=0             # เลือก GPU หมายเลข 0 (หรือใช้ 'cpu' ถ้าไม่มี GPU)
)