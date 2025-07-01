import glob
import os
import re
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
from ultralytics import YOLO
import json
import base64
from datetime import datetime
import uvicorn
from segment_KyTu import test_license_plate as cnn_test_license_plate
import tempfile
import asyncio

app = FastAPI()

# Tải mô hình YOLO
MODEL_PATH = './model/yolov8n_trained.pt'
model = YOLO(MODEL_PATH)
CLASS_NAMES = ['bien_so']
JSON_OUTPUT_REALTIME = 'data/license_plate_realtime.json'
JSON_OUTPUT_VIDEO = 'data/License_Plate_Video.json'  # JSON path mới cho video

# Hàm kiểm tra biển số hợp lệ
def bien_so_hop_le(license_plate):
    if not license_plate or not isinstance(license_plate, str):
        return False
    if len(license_plate) < 8 or len(license_plate) > 10:
        return False
    pattern1 = r'^\d{2}[A-Z]{2}\d{5}$'  # 2 chữ hoa + 5 số
    pattern2 = r'^\d{2}[A-Z]\d\d{4,5}$'  # 1 chữ hoa + 1 số + 4–5 số
    return re.match(pattern1, license_plate) or re.match(pattern2, license_plate)

# Hàm xử lý khung hình
async def process_frame(image_data: bytes):
    # Chuyển bytes thành ảnh
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return []

    # Tải JSON cũ (nếu có)
    results = []
    if os.path.exists(JSON_OUTPUT_REALTIME):
        try:
            with open(JSON_OUTPUT_REALTIME, 'r') as f:
                file_content = f.read().strip()
                results = json.loads(file_content) if file_content else []
        except json.JSONDecodeError:
            results = []

    # Dự đoán với YOLO
    yolo_results = model.predict(image, conf=0.5, iou=0.45)
    print(len(yolo_results))
    # Danh sách các biển số đã có để kiểm tra trùng lặp
    existing_plates = {result["License_Plate"] for result in results}

    for result in yolo_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Cắt vùng biển số
            plate_img = image[y1:y2, x1:x2]
            # Nhận diện ký tự
            result = cnn_test_license_plate(plate_img)
            if result["status"] != "success" or not result["license_plate"]:
                continue
            license_plate_text = result["license_plate"]
            if not bien_so_hop_le(license_plate_text):
                continue

            # Kiểm tra trùng lặp biển số
            if license_plate_text in existing_plates:
                continue  # Bỏ qua nếu biển số đã tồn tại

            # Chuyển ảnh thành base64
            _, buffer = cv2.imencode('.jpg', plate_img)
            base64_str = base64.b64encode(buffer).decode('utf-8')

            # Tạo dữ liệu kết quả
            result_data = {
                "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "License_Plate": license_plate_text,
                "Image": base64_str
            }
            results.insert(0, result_data)  # Thêm vào đầu danh sách
            existing_plates.add(license_plate_text)  # Cập nhật danh sách biển số

    # Ghi lại kết quả
    with open(JSON_OUTPUT_REALTIME, 'w') as f:
        json.dump(results, f, indent=4)

    return results

async def test_model(image_data: bytes, conf_threshold: float = 0.5, iou_threshold: float = 0.5):
    # Thư mục lưu ảnh debug
    debug_image_dir = "image"
    os.makedirs(debug_image_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Timestamp để đặt tên file duy nhất

    try:
        # Chuyển bytes thành hình ảnh
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # print(f"[DEBUG] Đã giải mã ảnh đầu vào, shape: {image.shape if image is not None else 'None'}")
        
        # Lưu ảnh đầu vào
        input_image_path = os.path.join(debug_image_dir, f"input_{timestamp}.jpg")
        if image is not None:
            cv2.imwrite(input_image_path, image)
            #print(f"[DEBUG] Đã lưu ảnh đầu vào tại {input_image_path}")
        else:
            raise ValueError("Không thể giải mã ảnh")

        yolo_results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
        #print(f"[DEBUG] Kết quả từ YOLO: {len(yolo_results)} vùng được phát hiện")
        license_plate_texts = []

        for i, result in enumerate(yolo_results):
            if result.boxes is None:
                # print(f"[DEBUG] Không có hộp nào được phát hiện trong kết quả {i}")
                continue
            for j, box in enumerate(result.boxes):
                coords = box.xyxy[0].detach().cpu().numpy().astype(int)
                x1, y1, x2, y2 = coords
                # print(f"[DEBUG] Hộp {j} tọa độ: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={float(box.conf[0]):.2f}")

                if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                    # print(f"[DEBUG] Tọa độ không hợp lệ: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    continue

                confidence = float(box.conf[0])
                plate_img = image[y1:y2, x1:x2]
                # print(f"[DEBUG] Hình ảnh biển số {j} shape: {plate_img.shape}")

                # Lưu ảnh vùng biển số
                plate_image_path = os.path.join(debug_image_dir, f"plate_{timestamp}_{i}_{j}.jpg")
                if plate_img.size > 0:
                    cv2.imwrite(plate_image_path, plate_img)
                    # print(f"[DEBUG] Đã lưu ảnh biển số tại {plate_image_path}")
                else:
                    print(f"[DEBUG] Hình ảnh biển số rỗng, bỏ qua...")
                    continue

                # Phóng to ảnh nếu cần
                min_size = 64
                if plate_img.shape[0] < min_size or plate_img.shape[1] < min_size:
                    scale = max(min_size / plate_img.shape[0], min_size / plate_img.shape[1])
                    new_size = (int(plate_img.shape[1] * scale), int(plate_img.shape[0] * scale))
                    plate_img = cv2.resize(plate_img, new_size, interpolation=cv2.INTER_CUBIC)
                    # print(f"[DEBUG] Đã phóng to ảnh biển số, shape mới: {plate_img.shape}")
                    # Lưu ảnh biển số sau khi phóng to
                    resized_plate_path = os.path.join(debug_image_dir, f"resized_plate_{timestamp}_{i}_{j}.jpg")
                    cv2.imwrite(resized_plate_path, plate_img)
                    # print(f"[DEBUG] Đã lưu ảnh biển số phóng to tại {resized_plate_path}")

                try:
                    result = cnn_test_license_plate(plate_img)
                    # print(f"[DEBUG] Kết quả từ cnn_test_license_plate: {result}")
                    if result["status"] == "success" and result["license_plate"]:
                        license_plate_text = result["license_plate"]
                        license_plate_texts.append(license_plate_text)
                        print(f"[DEBUG] Biển số nhận diện: {license_plate_text}")

                        # Vẽ khung và ký tự lên ảnh
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text_y = max(y1 - 10, 10)
                        cv2.putText(image, f"{license_plate_text} {confidence:.2f}", (x1, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print(f"[DEBUG] Không nhận diện được biển số: {result['message']}")
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        text_y = max(y1 - 10, 10)
                        cv2.putText(image, f"CNN {confidence:.2f}", (x1, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"[DEBUG] Lỗi trong cnn_test_license_plate: {str(e)}")
                    continue

        # Lưu ảnh kết quả sau khi vẽ khung
        output_image_path = os.path.join(debug_image_dir, f"output_{timestamp}.jpg")
        cv2.imwrite(output_image_path, image)
        # print(f"[DEBUG] Đã lưu ảnh kết quả tại {output_image_path}")

        # Mã hóa ảnh kết quả thành base64
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Không thể mã hóa ảnh")
        base64_str = base64.b64encode(buffer).decode('utf-8')
        # print(f"[DEBUG] Đã mã hóa ảnh kết quả thành base64")

        return {
            "License_Plate": license_plate_texts,
            "Image": base64_str,
        }
    except Exception as e:
        # print(f"[DEBUG] Lỗi trong test_model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")

# Endpoint xử lý video (sửa JSON path)
@app.post("/video/")
async def detect_license_plate_video(file: UploadFile = File(...)):
    try:
        # Tạo file tạm để lưu video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Mở video bằng OpenCV
        cap = cv2.VideoCapture(temp_file_path)
        if not cap.isOpened():
            os.unlink(temp_file_path)
            raise HTTPException(status_code=400, detail="Không thể mở video")

        # Lấy thông tin video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tải JSON cũ (nếu có)
        json_data = []
        if os.path.exists(JSON_OUTPUT_VIDEO):
            try:
                with open(JSON_OUTPUT_VIDEO, 'r') as f:
                    file_content = f.read().strip()
                    json_data = json.loads(file_content) if file_content else []
            except json.JSONDecodeError:
                json_data = []

        # Danh sách các biển số đã có để kiểm tra trùng lặp
        existing_plates = {result["License_Plate"] for result in json_data}

        # Danh sách lưu kết quả mới từ video
        results = []
        frame_count = 0
        processed_frames = 0
        frame_skip = 5  # Xử lý mỗi 5 frame để tối ưu hiệu suất

        # Đọc và xử lý từng frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            processed_frames += 1

            # Chuyển frame thành bytes để xử lý
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue
            image_data = buffer.tobytes()

            # Xử lý frame bằng test_model
            try:
                result = await test_model(image_data)
                if result["License_Plate"]:
                    for license_plate in result["License_Plate"]:
                        # Kiểm tra trùng lặp biển số
                        if license_plate in existing_plates:
                            continue  # Bỏ qua nếu biển số đã tồn tại

                        result_data = {
                            "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "Frame": frame_count,
                            "License_Plate": license_plate,
                            "Image": result["Image"]
                        }
                        results.append(result_data)
                        existing_plates.add(license_plate)  # Cập nhật danh sách biển số
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                continue

        # Giải phóng video và xóa file tạm
        cap.release()
        os.unlink(temp_file_path)

        # Thêm kết quả mới vào JSON
        json_data.extend(results)
        os.makedirs(os.path.dirname(JSON_OUTPUT_VIDEO), exist_ok=True)
        with open(JSON_OUTPUT_VIDEO, 'w') as f:
            json.dump(json_data, f, indent=4)

        return {
            "message": f"Processed {processed_frames} frames",
            "results": results
        }
    except Exception as e:
        # Xóa file tạm nếu có lỗi
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý video: {str(e)}")

# WebSocket endpoint để xử lý video thời gian thực
@app.websocket("/ws/video")
async def websocket_video_endpoint(websocket: WebSocket):
    json_dir = "data/License_Plate_Video.json"  # Thay bằng đường dẫn thực tế
    output_image_dir = "image"  # Thư mục để lưu hình ảnh

    # Tạo thư mục lưu hình ảnh nếu chưa tồn tại
    os.makedirs(output_image_dir, exist_ok=True)

    # Xóa tất cả file JSON trong thư mục (nếu cần)
    try:
        with open(json_dir, 'w') as f:
            json.dump([], f)
        # print(f"[DEBUG] Đã làm rỗng file JSON: {json_dir}")
    except Exception as e:
        print(f"[DEBUG] Lỗi khi làm rỗng file JSON: {e}")

    await websocket.accept()
    frame_count = 0
    processed_frames = 0
    frame_skip = 5  # Xử lý mỗi 5 frame để tối ưu hiệu suất
    results = []

    # Tải JSON cũ (nếu có) để kiểm tra trùng lặp
    json_data = []
    if os.path.exists(json_dir):
        try:
            with open(json_dir, 'r') as f:
                file_content = f.read().strip()
                json_data = json.loads(file_content) if file_content else []
            # print(f"[DEBUG] Đã tải dữ liệu từ file JSON: {len(json_data)} bản ghi")
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Lỗi khi đọc file JSON: {e}")
            json_data = []

    # Danh sách các biển số đã có để kiểm tra trùng lặp
    existing_plates = {result["License_Plate"] for result in json_data}
    # print(f"[DEBUG] Số biển số đã có: {len(existing_plates)}")

    try:
        while True:
            # Nhận dữ liệu frame từ client (bytes)
            image_data = await websocket.receive_bytes()
            frame_count += 1
            # print(f"[DEBUG] Nhận frame {frame_count}")

            if frame_count % frame_skip != 0:
                # print(f"[DEBUG] Bỏ qua frame {frame_count}")
                continue

            processed_frames += 1
            # print(f"[DEBUG] Xử lý frame {frame_count} (Processed: {processed_frames})")

            # Lưu hình ảnh để kiểm tra
            try:
                # Chuyển bytes thành hình ảnh bằng OpenCV
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    image_path = os.path.join(output_image_dir, f"frame_{frame_count}.jpg")
                    cv2.imwrite(image_path, img)
                    # print(f"[DEBUG] Đã lưu hình ảnh frame {frame_count} tại {image_path}")
                else:
                    print(f"[DEBUG] Lỗi: Không thể giải mã hình ảnh frame {frame_count}")
            except Exception as e:
                print(f"[DEBUG] Lỗi khi lưu hình ảnh frame {frame_count}: {e}")

            # Xử lý frame bằng test_model
            try:
                result = await test_model(image_data)
                #print(f"[DEBUG] Kết quả từ test_model cho frame {frame_count}: {result}")
                
                if result["License_Plate"]:
                    for license_plate in result["License_Plate"]:
                        # Kiểm tra trùng lặp biển số
                        if license_plate in existing_plates:
                            #print(f"[DEBUG] Biển số {license_plate} đã tồn tại, bỏ qua")
                            continue

                        result_data = {
                            "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "Frame": frame_count,
                            "License_Plate": license_plate,
                            "Image": result["Image"]
                        }
                        results.append(result_data)
                        existing_plates.add(license_plate)
                        #print(f"[DEBUG] Thêm biển số mới: {license_plate}")

                        # Gửi kết quả về client qua WebSocket
                        await websocket.send_json(result_data)
                        #print(f"[DEBUG] Đã gửi dữ liệu về client: {result_data}")

                # Lưu kết quả vào JSON
                json_data.extend(results)
                os.makedirs(os.path.dirname(json_dir), exist_ok=True)
                with open(json_dir, 'w') as f:
                    json.dump(json_data, f, indent=4)
                #print(f"[DEBUG] Đã lưu {len(results)} kết quả vào JSON")

                # Reset results để tránh lặp lại dữ liệu trong vòng lặp
                results = []

            except Exception as e:
                print(f"[DEBUG] Lỗi xử lý frame {frame_count}: {str(e)}")
                await websocket.send_json({"error": f"Lỗi xử lý frame: {str(e)}"})
                continue

    except Exception as e:
        print(f"[DEBUG] Lỗi WebSocket: {str(e)}")
        await websocket.send_json({"error": f"Lỗi WebSocket: {str(e)}"})
    finally:
        print(f"[DEBUG] Đóng kết nối WebSocket")
        await websocket.close()
        
# Endpoint nhận khung hình
@app.post("/upload-frame/")
async def upload_frame(file: UploadFile = File(...)):
    image_data = await file.read()
    results = await process_frame(image_data)
    return {"results": results}

# Endpoint lấy file JSON
@app.get("/get-all-results-realtime/")
async def get_results():
    if os.path.exists(JSON_OUTPUT_REALTIME):
        return FileResponse(JSON_OUTPUT_REALTIME)
    return {"error": "No results available"}

# Endpoint lấy file JSON lịch sử
@app.get("/get-all-results-history/")
async def get_results():
    if os.path.exists("data/License_Plate.json"):
        return FileResponse("data/License_Plate.json")
    return {"error": "No results available"}

# Endpoint lấy file JSON video
@app.get("/get-all-results-video/")
async def get_video_results():
    if os.path.exists(JSON_OUTPUT_VIDEO):
        return FileResponse(JSON_OUTPUT_VIDEO)
    return {"error": "No video results available"}

# WebSocket để gửi cập nhật thời gian thực cho ảnh
@app.websocket("/ws/results")
async def websocket_endpoint(websocket: WebSocket):
    # json_file = JSON_OUTPUT_REALTIME  # Đường dẫn tới file JSON
    # try:
    #     # Ghi đè file JSON với mảng rỗng để làm rỗng dữ liệuN
    #     with open(json_file, 'w') as f:
    #         json.dump([], f)
    #     print(f"Đã làm rỗng file JSON: {json_file}")
    # except Exception as e:
    #     print(f"Lỗi khi làm rỗng file JSON: {e}")

    await websocket.accept()
    while True:
        try:
            image_data = await websocket.receive_bytes()
            results = await process_frame(image_data)
            await websocket.send_json(results)
        except Exception as e:
            print(f"Lỗi WebSocket: {e}")
            break

@app.post("/image/")
async def detect_license_plate(file: UploadFile = File(...)):
    image_data = await file.read()
    # print(f"Filename: {file.filename}")
    # print(f"Content type: {file.content_type}")
    # print(f"Type of image_data: {type(image_data)}")  # <class 'bytes'>

    try:
        result = await test_model(image_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)