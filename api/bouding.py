from ultralytics import YOLO
import cv2
import numpy as np
from segment_KyTu import test_license_plate as cnn_test_license_plate

# Nếu bạn có tên lớp, thêm ở đây
CLASS_NAMES = ['bien_so']

def test_model(image_path, yolo_model_path='./model/yolov8n_trained.pt', output_path='output_annotated.jpg', conf_threshold=0.5, iou_threshold=0.45):
    """
    Chạy mô hình YOLOv8 để phát hiện biển số và tích hợp với CNN để nhận diện ký tự.
    Args:
        image_path (str): Đường dẫn ảnh đầu vào.
        yolo_model_path (str): Đường dẫn đến mô hình YOLOv8 đã huấn luyện.
        output_path (str): Đường dẫn ảnh đầu ra sau khi vẽ bounding boxes.
        conf_threshold (float): Ngưỡng độ tin cậy tối thiểu.
        iou_threshold (float): Ngưỡng IOU cho Non-Max Suppression.
    """
    # Tải mô hình YOLOv8 đã huấn luyện
    model = YOLO(yolo_model_path)

    # Đọc ảnh đầu vào
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image at {image_path}")
        return

    # Dự đoán đối tượng trong ảnh
    results = model.predict(image, conf=conf_threshold, iou=iou_threshold)

    total_boxes = 0  # Đếm số khung dự đoán

    # Lặp qua tất cả các kết quả và vẽ các khung bao quanh biển số
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ID {class_id}"

            # Cắt vùng biển số
            plate_img = image[y1:y2, x1:x2]
            plate_img_path = 'temp_plate.jpg'
            cv2.imwrite(plate_img_path, plate_img)

            # Gọi CNN để nhận diện ký tự
            license_plate_text = cnn_test_license_plate(plate_img_path, x1, y1, x2, y2)

            # Vẽ khung dự đoán và hiển thị kết quả CNN
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{license_plate_text} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"[DETECTION] Class: {class_name}, Confidence: {confidence:.2f}, "
                  f"Box: ({x1}, {y1}, {x2}, {y2}), License Plate: {license_plate_text}")

            total_boxes += 1

    print(f"[SUMMARY] Total detections: {total_boxes}")

    # Lưu ảnh đã annotate
    cv2.imwrite(output_path, image)
    print(f"[INFO] Annotated image saved at: {output_path}")

