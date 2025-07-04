# AI NhanDienBienSoXe

## Overview / Tổng quan

- **English**:  
  This is an AI system built to recognize vehicle license plates using images or videos from cameras. It uses two CNN models: YOLO for detecting license plates and EfficientNet for identifying characters on the plates. The system works well in good weather conditions. It detects license plates, segments characters, and recognizes them as numbers or letters. The final output includes a bounding box around the license plate and the license plate information.

- **Tiếng Việt**:  
  Đây là một hệ thống AI được xây dựng để nhận diện biển số xe thông qua hình ảnh hoặc video từ camera. Hệ thống sử dụng hai mô hình CNN: YOLO để phát hiện biển số xe và EfficientNet để nhận diện các ký tự trên biển số. AI hoạt động tốt trong điều kiện thời tiết thuận lợi. Nó phát hiện biển số, phân đoạn các ký tự, và nhận diện chúng là số hoặc chữ. Kết quả cuối cùng bao gồm một khung bao quanh biển số và thông tin chi tiết của biển số đó.

## Technologies and Libraries / Công nghệ và thư viện

- **English**:  
  - **YOLOv8**: Used for detecting license plates in images or videos.  
  - **EfficientNet**: Used for recognizing characters on the license plates.  
  - **FastAPI**: Provides a web API to process images and videos.  
  - **OpenCV**: Handles image and video processing.  
  - **PyTorch**: Supports the deep learning models.  
  - **Roboflow**: Used for character detection in license plates.  
  - **Python Libraries**: NumPy, Matplotlib, PIL, and others for data processing and visualization.  

- **Tiếng Việt**:  
  - **YOLOv8**: Sử dụng để phát hiện biển số xe trong hình ảnh hoặc video.  
  - **EfficientNet**: Dùng để nhận diện các ký tự trên biển số xe.  
  - **FastAPI**: Cung cấp API web để xử lý hình ảnh và video.  
  - **OpenCV**: Xử lý hình ảnh và video.  
  - **PyTorch**: Hỗ trợ các mô hình học sâu.  
  - **Roboflow**: Sử dụng để phát hiện ký tự trên biển số.  
  - **Thư viện Python**: NumPy, Matplotlib, PIL và các thư viện khác để xử lý và trực quan hóa dữ liệu.

## Installation / Hướng dẫn cài đặt

- **English**:  
  Follow these steps to set up the project on your computer:  

  1. **Clone the Repository**:  
     ```bash
     git clone https://github.com/your-username/AI-NhanDienBienSoXe.git
     cd AI-NhanDienBienSoXe
     ```

  2. **Install Python**:  
     Make sure you have Python 3.8 or higher installed. You can download it from [python.org](https://www.python.org/).

  3. **Create a Virtual Environment** (optional but recommended):  
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```

  4. **Install Required Libraries**:  
     Install the necessary Python libraries using the `requirements.txt` file:  
     ```bash
     pip install -r requirements.txt
     ```

  5. **Download Pre-trained Models**:  
     - Download the YOLOv8 model (`yolov8n_trained.pt`) and place it in the `model/` folder.  
     - Download the EfficientNet model (`character_cnn_final.pth`) and place it in the `model/` folder.  

  6. **Run the Application**:  
     Start the FastAPI server to process images or videos:  
     ```bash
     python api.py
     ```
     The server will run at `http://0.0.0.0:9999`. You can access the API endpoints to test the system.

- **Tiếng Việt**:  
  Thực hiện các bước sau để cài đặt dự án trên máy tính của bạn:  

  1. **Tải mã nguồn về máy**:  
     ```bash
     git clone https://github.com/your-username/AI-NhanDienBienSoXe.git
     cd AI-NhanDienBienSoXe
     ```

  2. **Cài đặt Python**:  
     Đảm bảo bạn đã cài đặt Python phiên bản 3.8 hoặc cao hơn. Tải Python tại [python.org](https://www.python.org/).

  3. **Tạo môi trường ảo** (khuyến nghị nhưng không bắt buộc):  
     ```bash
     python -m venv venv
     source venv/bin/activate  # Trên Windows: venv\Scripts\activate
     ```

  4. **Cài đặt các thư viện cần thiết**:  
     Cài đặt các thư viện Python cần thiết bằng file `requirements.txt`:  
     ```bash
     pip install -r requirements.txt
     ```

  5. **Tải các mô hình đã huấn luyện**:  
     - Tải mô hình YOLOv8 (`yolov8n_trained.pt`) và đặt vào thư mục `model/`.  
     - Tải mô hình EfficientNet (`character_cnn_final.pth`) và đặt vào thư mục `model/`.  

  6. **Chạy ứng dụng**:  
     Khởi động server FastAPI để xử lý hình ảnh hoặc video:  
     ```bash
     python api.py
     ```
     Server sẽ chạy tại `http://0.0.0.0:9999`. Bạn có thể truy cập các endpoint API để kiểm tra hệ thống.

## Usage / Cách sử dụng

- **English**:  
  - **Upload an Image**: Use the `/image/` endpoint to upload an image and get the license plate information.  
  - **Upload a Video**: Use the `/video/` endpoint to process a video and detect license plates in multiple frames.  
  - **Real-time Processing**: Use the `/ws/video` WebSocket endpoint for real-time license plate detection from a camera feed.  
  - **View Results**: Access the JSON results at `/get-all-results-realtime/` or `/get-all-results-video/` to see the detected license plates and their images in base64 format.

- **Tiếng Việt**:  
  - **Tải lên hình ảnh**: Sử dụng endpoint `/image/` để tải lên một hình ảnh và nhận thông tin biển số xe.  
  - **Tải lên video**: Sử dụng endpoint `/video/` để xử lý video và phát hiện biển số xe trong nhiều khung hình.  
  - **Xử lý thời gian thực**: Sử dụng endpoint WebSocket `/ws/video` để nhận diện biển số xe thời gian thực từ nguồn camera.  
  - **Xem kết quả**: Truy cập kết quả JSON tại `/get-all-results-realtime/` hoặc `/get-all-results-video/` để xem danh sách biển số xe đã phát hiện và hình ảnh của chúng ở định dạng base64.

## Examples / Ví dụ

- **English**:  
  Below are examples of input and output images to show how the system detects and recognizes license plates. The input image is the original photo, and the output image includes a bounding box around the license plate with the recognized text.

  - **Layout mobile input Image**: Original image from a camera.  
    ![Input Image](image/Picture3.png)  
  - **Layout mobile output Image**: Image with detected license plate and text.  
    ![Output Image](image/Picture4.png)
  - **Layout web**: Image with detected license plate and text.  
    ![Output Image](image/Picture2.png)

- **Tiếng Việt**:  
  Dưới đây là các ví dụ về hình ảnh đầu vào và đầu ra để minh họa cách hệ thống phát hiện và nhận diện biển số xe. Hình ảnh đầu vào là ảnh gốc từ camera, và hình ảnh đầu ra bao gồm khung bao quanh biển số cùng với văn bản được nhận diện.

  - **Giao diện hình ảnh đầu vào**: Ảnh gốc từ camera.  
    ![Hình ảnh đầu vào](image/Picture3.png)  
  - **Giao diện hình ảnh đầu ra**: Ảnh với biển số được phát hiện và văn bản.  
    ![Hình ảnh đầu ra](image/Picture4.png)
  - **Giao diện web**: Ảnh với biển số được phát hiện và văn bản.  
    ![Output Image](image/Picture2.png)


## Notes / Lưu ý

- **English**:  
  - The system works best in clear weather and good lighting conditions.  
  - Ensure the camera resolution is high enough for accurate detection.  
  - The license plate format must match the patterns defined in the code (e.g., `12AB12345` or `12A11234`).  
  - For any issues, check the debug logs in the `image/` folder or the console output.

- **Tiếng Việt**:  
  - Hệ thống hoạt động tốt nhất trong điều kiện thời tiết rõ ràng và ánh sáng tốt.  
  - Đảm bảo độ phân giải camera đủ cao để nhận diện chính xác.  
  - Định dạng biển số xe phải khớp với các mẫu được định nghĩa trong mã (ví dụ: `12AB12345` hoặc `12A11234`).  
  - Nếu gặp sự cố, kiểm tra nhật ký gỡ lỗi trong thư mục `image/` hoặc đầu ra trên console.

