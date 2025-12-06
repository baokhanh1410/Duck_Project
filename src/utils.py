import cv2

CLASS_COLORS = {
    0: (0, 255, 0),      # Xanh lá
    1: (0, 200, 255),    # Xanh ngọc
    2: (255, 255, 0),    # Xanh dương nhạt
    3: (0, 128, 255),    # Cam
    4: (0, 0, 255),      # Đỏ
    5: (180, 0, 180),    # Tím
    6: (255, 0, 255),    # Hồng
    7: (128, 128, 128)   # Xám
}

CLASS_NAMES = {
    0: "Khoe-Dung",
    1: "Khoe-Ngoi",
    2: "Khoe-Nam",
    3: "Khoe-Khac",
    4: "Yeu-Dung",
    5: "Yeu-Ngoi",
    6: "Yeu-Nam",
    7: "Yeu-Khac"
}

def draw_tracks(frame, tracks):
    for bbox, track_id, label in tracks:
        color = CLASS_COLORS.get(label, (255, 255, 255))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cls = CLASS_NAMES.get(label, "Unknown")
        text = f"ID:{track_id} {cls}"
        font_scale = 0.4
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
    return frame
