# 🎭 Nhận Diện Cảm Xúc Khuôn Mặt & Gợi Ý Nhạc (Facial Emotion Recognition & Spotify Integration)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)
[![Spotify API](https://img.shields.io/badge/Spotify-API-1DB954.svg)](https://developer.spotify.com/)

Dự án này ứng dụng **Deep Learning (Học Sâu)** vào bài toán nhận diện cảm xúc khuôn mặt theo thời gian thực, từ đó tự động gợi ý và phát các bài hát trên nền tảng **Spotify** phù hợp với tâm trạng hiện tại của người dùng.

---

## 🌟 1. Giới thiệu ứng dụng

Âm nhạc và cảm xúc có một sự liên kết chặt chẽ. Ứng dụng này ra đời nhằm mục đích mang lại một trải nghiệm âm nhạc cá nhân hóa hoàn toàn tự động. Thay vì phải tự tìm kiếm một bài hát phù hợp với tâm trạng, hệ thống sẽ tự động "đọc" biểu cảm trên khuôn mặt bạn thông qua Webcam và yêu cầu Spotify phát một danh sách phát (playlist) tương ứng.

**Các tính năng nổi bật:**
* Nhận diện khuôn mặt và phân loại cảm xúc theo thời gian thực.
* Sử dụng mô hình tiên tiến **Swin Transformer** cho độ chính xác cao trong nhận diện ảnh.
* Tự động kết nối và điều khiển trình phát nhạc Spotify của người dùng.

---

## 🏗️ 2. Kiến trúc hệ thống

Hệ thống được chia thành 3 luồng xử lý chính:

1.  **Data Capture (Thu thập dữ liệu):** Camera chụp luồng hình ảnh khuôn mặt của người dùng theo thời gian thực (thường sử dụng thư viện OpenCV).
2.  **Emotion Classifier (Phân loại cảm xúc - AI Layer):** Hình ảnh sau khi được tiền xử lý sẽ được đưa vào mô hình Deep Learning (**Swin Transformer**). Mô hình tiến hành phân tích các đặc trưng trên khuôn mặt và trả về kết quả dự đoán cảm xúc (Ví dụ: Vui, Buồn, Tức giận, Bình thường...).
3.  **Music Recommendation (Tích hợp & Gợi ý - Application Layer):** Kết quả cảm xúc được ánh xạ (mapping) tới một Spotify Playlist ID cụ thể. Hệ thống gọi Spotify Web API thông qua `spotipy` để kích hoạt phát nhạc trên thiết bị của người dùng.

---

## 💻 3. Công nghệ sử dụng

Dự án sử dụng các công nghệ, ngôn ngữ và thư viện sau:

* **Ngôn ngữ lập trình:** Python.
* **Deep Learning Framework:** PyTorch (hỗ trợ tính toán song song với CUDA GPU).
* **Mô hình thị giác máy tính (Computer Vision):** Swin Transformer (Shifted Window Hierarchical Vision Transformer) - Tối ưu cho các bài toán phân loại ảnh.
* **Xử lý hình ảnh:** OpenCV (`cv2`).
* **Giao tiếp API:** Thư viện `spotipy` (để giao tiếp với Spotify Web API).
* **Quản lý biến môi trường:** `python-dotenv` (đọc file `.env`).

---

## ⚙️ 4. Nghiệp vụ cốt lõi

* **Xử lý Mô hình (Swin Transformer):** File `swin-trans.py` và `test_swin.py` định nghĩa cấu trúc và cách tải trọng số (weights) của mô hình. Mô hình phân chia hình ảnh thành các "patch" nhỏ và áp dụng cơ chế Self-Attention để bắt các đặc điểm vi mô thể hiện cảm xúc trên cơ mặt.
* **Mapping Cảm xúc & Âm nhạc (`play_emotion.py`):** Xây dựng bộ quy tắc (rule-based) để chuyển đổi từ `Emotion Label` (Nhãn cảm xúc) sang `Spotify URI`.
    * *Happy (Vui vẻ)* ➡️ Playlist nhạc Pop, EDM sôi động.
    * *Sad (Buồn bã)* ➡️ Playlist nhạc Acoustic, Ballad nhẹ nhàng, chill.
    * *Angry (Tức giận)* ➡️ Playlist nhạc Rock hoặc nhạc thư giãn thiền định (tùy cấu hình).
* **Quản lý phiên Spotify (OAuth 2.0):** Xác thực người dùng qua API của Spotify để lấy Access Token, từ đó cấp quyền cho ứng dụng có thể đọc trạng thái player và điều khiển phát nhạc.

---

## 🚀 5. Hướng dẫn cài đặt và khởi chạy

### Yêu cầu trước khi cài đặt:
* Đã cài đặt Python 3.8 trở lên.
* Có thiết bị Webcam hoạt động.
* Có tài khoản **Spotify Premium** (Bắt buộc để sử dụng tính năng điều khiển Playback của Spotify API).

### Bước 1: Clone Repository
```bash
git clone [https://github.com/mut2407/DEEP-LEARNING-APPLICATIONS-IN-FACIAL-RECOGNITION.git](https://github.com/mut2407/DEEP-LEARNING-APPLICATIONS-IN-FACIAL-RECOGNITION.git)
cd DEEP-LEARNING-APPLICATIONS-IN-FACIAL-RECOGNITION
```

### Bước 2: Tạo môi trường ảo
- Kích hoạt môi trường ảo trên Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Bước 3: Khởi chạy ứng dụng
```bash
python run_project.py
# hoặc
python play_emotion.py
```