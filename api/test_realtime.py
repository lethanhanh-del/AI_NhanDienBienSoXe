import cv2
from ultralytics import YOLO
from segment_KyTu import test_license_plate as cnn_test_license_plate
import threading
import queue
import time
import numpy as np

# Hàng đợi
plate_queue = queue.Queue()
result_queue = queue.Queue()
stop_thread = False

def iou(box1, box2):
    """Tính IoU để theo dõi biển số."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def cnn_processor():
    """Luồng xử lý CNN."""
    while not stop_thread:
        plates = []
        coords = []
        # Gom batch tối đa 8 vùng
        while len(plates) < 8 and not plate_queue.empty():
            plate_img, x1, y1, x2, y2 = plate_queue.get_nowait()
            plates.append(plate_img)
            coords.append((x1, y1, x2, y2))
        
        if plates:
            for plate_img, (x1, y1, x2, y2) in zip(plates, coords):
                text = cnn_test_license_plate(plate_img, x1, y1, x2, y2)
                result_queue.put((text, x1, y1, x2, y2))

def detect_license_plate_realtime(model_path='./model/yolov8n_trained.pt'):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow('Real-time License Plate Detection', cv2.WINDOW_NORMAL)

    cnn_thread = threading.Thread(target=cnn_processor)
    cnn_thread.start()

    last_results = {}
    tracked_plates = {}
    plate_id = 0
    frame_count = 0
    last_boxes = []

    while True:
        start_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Khung hình không hợp lệ, bỏ qua...")
            continue

        frame = cv2.resize(frame, (416, 416))

        # Chạy YOLO mỗi 2 khung hình
        if frame_count % 2 == 0:
            results = model.predict(frame, conf=0.3, iou=0.5)
            last_boxes = results
        else:
            results = last_boxes
        frame_count += 1

        # Theo dõi và thêm biển số mới
        active_plates = set()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                plate_img = frame[y1:y2, x1:x2]
                if plate_img.size == 0:
                    print("Vùng biển số rỗng, bỏ qua...")
                    continue

                new_plate = True
                for pid, (px1, py1, px2, py2) in tracked_plates.items():
                    if iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
                        new_plate = False
                        tracked_plates[pid] = (x1, y1, x2, y2)
                        active_plates.add(pid)
                        break

                if new_plate and plate_queue.qsize() < 50:
                    plate_id += 1
                    tracked_plates[plate_id] = (x1, y1, x2, y2)
                    plate_queue.put((plate_img, x1, y1, x2, y2))

        # Dọn dẹp tracked_plates
        tracked_plates = {pid: box for pid, box in tracked_plates.items() if pid in active_plates}

        # Lấy kết quả từ CNN
        while not result_queue.empty():
            text, x1, y1, x2, y2 = result_queue.get()
            for pid, (px1, py1, px2, py2) in tracked_plates.items():
                if iou((x1, y1, x2, y2), (px1, py1, px2, py2)) > 0.5:
                    last_results[pid] = (text, confidence, (x1, y1, x2, y2))
                    break

        # Vẽ kết quả
        for pid, (text, confidence, (x1, y1, x2, y2)) in last_results.items():
            label = f"{text} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Tính và hiển thị FPS
        elapsed_time = time.perf_counter() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else float('inf')
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print("Trước khi hiển thị khung hình")
        cv2.imshow('Real-time License Plate Detection', frame)
        print("Sau khi hiển thị khung hình")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp khi thoát
    global stop_thread
    stop_thread = True
    cnn_thread.join()
    cap.release()
    cv2.destroyAllWindows()
