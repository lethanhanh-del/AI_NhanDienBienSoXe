import base64
from datetime import datetime
import json
import re
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from inference_sdk import InferenceHTTPClient

from model import CharacterCNN

# Khởi tạo client Roboflow
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="a92qVk3vE0o0PJhI3Swj"
)

# Danh sách nhãn
classes = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]  # 0-9, A-Z

# Chuẩn bị transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load mô hình đã huấn luyện
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterCNN(num_classes=36).to(device)
model.load_state_dict(torch.load("model/character_cnn_final.pth", weights_only=True))
model.eval()

# Hàm dự đoán một ảnh ký tự
def predict_character(char_img):
    try:
        # Đảm bảo ảnh là uint8
        if char_img.dtype != np.uint8:
            char_img = char_img.astype(np.uint8)
        
        # Chuyển ảnh sang PIL Image
        img = Image.fromarray(char_img)
        img = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            # Đảm bảo predicted là scalar
            if predicted.numel() != 1:
                print(f"Error: Predicted tensor has size {predicted.numel()}")
                return None, 0.0
                
            predicted_idx = predicted.item()
            if predicted_idx >= len(classes):
                print(f"Error: Predicted index {predicted_idx} out of range")
                return None, 0.0
                
            predicted_char = classes[predicted_idx]
            confidence = probabilities[0][predicted_idx].item()
            return predicted_char, confidence
    except Exception as e:
        print(f"Error in predict_character: {str(e)}")
        return None, 0.0

# Hàm tự động phát hiện loại biển số
def detect_plate_type(predictions, height_threshold=20, y_gap_threshold=30):
    if not predictions or len(predictions) < 2:
        return False
    
    y_coords = [pred['y'] for pred in predictions if 'y' in pred]
    if len(y_coords) < 2:
        return False
    
    mean_y = np.mean(y_coords)
    std_y = np.std(y_coords)
    
    if std_y > height_threshold:
        y_coords_sorted = sorted(y_coords)
        max_gap = max(y_coords_sorted[i] - y_coords_sorted[i-1] for i in range(1, len(y_coords_sorted)))
        if max_gap > y_gap_threshold:
            return True
    
    return False

# Hàm tính IoU giữa hai bounding box
def iou(box1, box2):
    x1_min = box1['x'] - box1['width'] / 2
    x1_max = box1['x'] + box1['width'] / 2
    y1_min = box1['y'] - box1['height'] / 2
    y1_max = box1['y'] + box1['height'] / 2
    
    x2_min = box2['x'] - box2['width'] / 2
    x2_max = box2['x'] + box2['width'] / 2
    y2_min = box2['y'] - box2['height'] / 2
    y2_max = box2['y'] + box2['height'] / 2
    
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = box1['width'] * box1['height']
    box2_area = box2['width'] * box2['height']
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# Hàm lọc các bounding box trùng lặp
def filter_overlapping_boxes(predictions, iou_threshold=0.5):
    if len(predictions) <= 1:
        return predictions
    
    filtered = []
    while predictions:
        best_box = max(predictions, key=lambda x: x['confidence'])
        filtered.append(best_box)
        predictions = [p for p in predictions if iou(best_box, p) <= iou_threshold]
    
    return filtered

def bien_so_hop_le(license_plate):
    if not license_plate or not isinstance(license_plate, str):
        return False
    if len(license_plate) < 7 or len(license_plate) > 9:
        return False
    pattern1 = r'^\d{2}[A-Z]{2}\d{5}$'  # 2 chữ hoa + 5 số
    pattern2 = r'^\d{2}[A-Z]\d\d{4,5}$'  # 1 chữ hoa + 1 số + 4–5 số
    return re.match(pattern1, license_plate) or re.match(pattern2, license_plate)

# Hàm lưu vào file JSON
def save_to_json(plate_img, license_plate, json_path='./data/License_Plate.json'):
    try:
        # Kiểm tra plate_img là mảng NumPy
        if not isinstance(plate_img, np.ndarray):
            print(f"Error in save_to_json: plate_img must be a NumPy array, got {type(plate_img)}")
            return False

        # Chuyển mảng NumPy thành base64
        _, buffer = cv2.imencode('.jpg', plate_img)
        if not buffer.any():
            print("Error in save_to_json: Failed to encode image")
            return False
        base64_str = base64.b64encode(buffer).decode('utf-8')

        json_data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as json_file:
                    json_data = json.load(json_file)
            except json.JSONDecodeError:
                print(f"Error in save_to_json: Invalid JSON in {json_path}")
                json_data = []

        record = {
            "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "License_Plate": license_plate,
            "Image": base64_str,
        }
        json_data.append(record)

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error in save_to_json: {str(e)}")
        return False
    
def sort_predictions(predictions, y_threshold=15):
    """
    Sắp xếp ký tự theo thứ tự từ trên xuống dưới, từ trái sang phải.
    :param predictions: list các dict có key 'x', 'y'
    :param y_threshold: ngưỡng sai số cho việc nhóm các dòng (tùy độ phân giải)
    :return: list đã sắp xếp
    """
    # B1: Nhóm các ký tự thành từng dòng dựa vào giá trị y
    lines = []
    for p in predictions:
        matched_line = False
        for line in lines:
            if abs(p['y'] - line[0]['y']) < y_threshold:
                line.append(p)
                matched_line = True
                break
        if not matched_line:
            lines.append([p])

    # B2: Sắp xếp các dòng theo y tăng dần (trên xuống dưới)
    lines = sorted(lines, key=lambda line: line[0]['y'])

    # B3: Sắp xếp các ký tự trong từng dòng theo x (trái sang phải)
    sorted_predictions = []
    for line in lines:
        sorted_predictions.extend(sorted(line, key=lambda p: p['x']))

    return sorted_predictions


# Hàm chính xử lý biển số
def test_license_plate(plate_img):
    # Initialize response
    response = {"status": "error", "message": "", "license_plate": ""}

    # Kiểm tra đầu vào là mảng NumPy
    if not isinstance(plate_img, np.ndarray):
        response["message"] = f"Đầu vào phải là mảng NumPy, got {type(plate_img)}"
        return response

    # Chuyển ảnh sang grayscale nếu cần
    if len(plate_img.shape) == 3:
        img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        img = plate_img

    if img is None or img.size == 0:
        response["message"] = "Ảnh không hợp lệ"
        return response

    # Phóng to ảnh nếu kích thước quá nhỏ
    min_size = 64
    if img.shape[0] < min_size or img.shape[1] < min_size:
        scale = max(min_size / img.shape[0], min_size / img.shape[1])
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
        # print(f"Resized plate_img to: {img.shape}")

    # Lưu ảnh tạm thời để gọi Roboflow
    temp_path = "temp_plate.jpg"
    cv2.imwrite(temp_path, img)

    # Gọi workflow Roboflow để nhận diện ký tự
    try:
        result = CLIENT.infer(temp_path, model_id="car-plate-characters-wcbkz-0eeoi/1")
        predictions = result.get('predictions', [])
        # print(f"Roboflow predictions: {predictions}")
    except Exception as e:
        response["message"] = f"Lỗi khi gọi infer: {str(e)}"
        os.remove(temp_path)  # Xóa file tạm
        return response

    # Xóa file tạm
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if not predictions:
        response["message"] = "Không tìm thấy ký tự nào trong ảnh"
        return response

    # Kiểm tra định dạng predictions
    required_keys = ['x', 'y', 'width', 'height']
    if not all(all(key in pred for key in required_keys) for pred in predictions):
        response["message"] = "Định dạng predictions không đúng"
        return response

    # Lọc các bounding box trùng lặp
    predictions = filter_overlapping_boxes(predictions)

    # Tự động phát hiện loại biển số
    is_motorcycle = detect_plate_type(predictions)

    # Sắp xếp predictions
    predictions = sort_predictions(predictions, y_threshold=15 if is_motorcycle else 5)


    # Xử lý các ký tự
    license_plate = ""
    char_index = 1  # Để đánh số thứ tự ký tự

    for pred in predictions:
        x = int(pred['x'] - pred['width'] / 2)
        y = int(pred['y'] - pred['height'] / 2)
        w = int(pred['width'])
        h = int(pred['height'])

        # Cắt ký tự
        char_img = img[y:y+h, x:x+w]
        if char_img.size == 0:
            continue

        char_img_resized = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)

        # # Hiển thị ảnh ký tự đã cắt
        # plt.figure(figsize=(1.5, 1.5))
        # plt.imshow(char_img_resized, cmap='gray')
        # plt.title(f'Char {char_index}')
        # plt.axis('off')
        # plt.show()
        # char_index += 1


        predicted_char, confidence = predict_character(char_img_resized)
        if predicted_char and confidence > 0.5:
            license_plate += predicted_char
            # print(f"Predicted char: {predicted_char}, confidence: {confidence}")

    if len(license_plate) < 7 or len(license_plate) > 9:
        response["message"] = f"Không Nhận Diện Được: Biển số {license_plate} có độ dài không hợp lệ"
        return response

    if not bien_so_hop_le(license_plate):
        response["message"] = f"Biển số {license_plate} không hợp lệ theo định dạng"
        return response

    # Lưu vào JSON và cập nhật response
    if license_plate:
        if save_to_json(plate_img, license_plate):
            response["status"] = "success"
            response["license_plate"] = license_plate
            response["message"] = "Xử lý thành công và lưu vào JSON"
        else:
            response["message"] = "Lỗi khi lưu vào JSON"
    else:
        response["message"] = "Không thể dự đoán biển số"

    return response

# Sử dụng
if __name__ == "__main__":
    images = [
        "29D163516.jpg",
        "30V3993.jpg"
    ]
    
    for image_path in images:
        result = test_license_plate(image_path)
        print(f"Xử lý ảnh {image_path}: {result}")