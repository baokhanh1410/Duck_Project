# Duck Behavior Analytics 🦆

Hệ thống Computer Vision giám sát sức khỏe đàn vịt, phân loại hành vi (Đứng, Ngồi, Nằm) và phân biệt nhóm Khỏe/Yếu.

## ⚠️ Lưu ý về Phạm vi Ứng dụng của Mô hình (Model Scope Disclaimer)

Mô hình **YOLOv12** được huấn luyện và tối ưu hóa **chuyên biệt** cho các điều kiện môi trường của dự án (góc quay cố định, điều kiện ánh sáng, mật độ vịt trong chuồng trại).

* **Mục tiêu huấn luyện:** Đảm bảo **độ ổn định và chính xác cao** cho bài toán phân tích hành vi và sức khỏe trong **bối cảnh dự án đã định**.
* **Hạn chế:** Khi thử nghiệm trên các hình ảnh hoặc video vịt có nguồn gốc khác (ví dụ: quay ngoài trời, góc quay không quen thuộc, điều kiện ánh sáng khác biệt), hiệu suất phát hiện (**Detection**) và theo dõi (**Tracking**) của mô hình **có thể bị giảm sút** do tính chất chuyên biệt của bộ dữ liệu huấn luyện.

## Tech Stack
- **Core:** Python, OpenCV
- **Model:** YOLOv12 (Detection) + ByteTrack (Tracking)
- **App:** Streamlit

## Installation

1. Clone repo:
    ```
    git clone https://github.com/baokhanh1410/Duck_Project.git
    cd Duck_Project
    ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

4. Run App:
    ```
    streamlit run app.py
    ```
