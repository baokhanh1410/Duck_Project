import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
import altair as alt

# Import 
from src.detector import Detector
from src.tracker import Tracker
from src.utils import draw_tracks, CLASS_NAMES

# Cấu hình trang
st.set_page_config(page_title="Duck Behavior Analytics", layout="wide")

st.title("🦆 Duck Behavior Monitoring System")
st.markdown("Hệ thống phát hiện, theo dõi và phân tích hành vi vịt sử dụng YOLOv12 & ByteTrack.")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Cấu hình Model")
model_conf = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# Đường dẫn model 
MODEL_PATH = "models/best.pt"

# --- MAIN APP ---
@st.cache_resource
def load_model():
    return Detector(model_path=MODEL_PATH)

try:
    detector = load_model()
    tracker = Tracker() # Khởi tạo tracker mới mỗi lần load lại
except Exception as e:
    st.error(f"Không tìm thấy model tại {MODEL_PATH}. Vui lòng kiểm tra lại folder 'models'.")
    st.stop()

uploaded_file = st.file_uploader("Upload Video Vịt (mp4, avi)", type=['mp4', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_frame = st.empty()
    
    with col2:
        st.markdown("### Thống kê Real-time")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st_count = st.empty()
        with kpi2:
            st_khoe = st.empty()
        with kpi3:
            st_yeu = st.empty()
            
        chart_placeholder = st.empty()

    data_stats = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # 1. Detect
        bboxes = detector.detect(frame, conf_threshold=model_conf)
        
        # 2. Track
        tracks = tracker.update(frame, bboxes)
        
        # 3. Draw
        frame_out = draw_tracks(frame.copy(), tracks)
        
        # 4. Thống kê 
        # Đếm số lượng hiện tại
        total_ducks = len(tracks)
        
        # Phân loại Khỏe/Yếu dựa trên class ID
        count_khoe = sum(1 for t in tracks if t[2] <= 3)
        count_yeu = sum(1 for t in tracks if t[2] >= 4)
        
        # Update KPIs
        st_count.metric("Tổng số", total_ducks)
        st_khoe.metric("Nhóm Khỏe", count_khoe)
        st_yeu.metric("Nhóm Yếu", count_yeu)

        # Lưu data để vẽ biểu đồ
        data_stats.append({
            "Frame": frame_count,
            "Total": total_ducks,
            "Healthy": count_khoe,
            "Weak": count_yeu
        })

        # Update Chart mỗi 5 frame
        if frame_count % 5 == 0:
            df = pd.DataFrame(data_stats)
            # Vẽ biểu đồ Line chart đơn giản
            chart = alt.Chart(df.tail(50)).mark_line().encode(
                x='Frame',
                y='Total',
                tooltip=['Frame', 'Total', 'Healthy', 'Weak']
            ).properties(height=200)
            chart_placeholder.altair_chart(chart, use_container_width=True)

        # Hiển thị Video
        # Convert BGR to RGB cho Streamlit
        frame_rgb = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    st.success("Đã xử lý xong video!")
