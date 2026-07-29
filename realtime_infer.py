import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import json
from collections import deque
from pathlib import Path
import sys
import io
from PIL import Image, ImageDraw, ImageFont

# --- Ép terminal dùng UTF-8 để in log tiếng Việt không lỗi ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CẤU HÌNH CƠ BẢN ---
K = 16
FEATURE_DIM = 132
L = 12
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng tin cậy tối thiểu để nhận ký hiệu

def load_json(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- HÀM VẼ TIẾNG VIỆT CÓ DẤU BẰNG PIL ---
def put_vietnamese_text(img, text, position, font_size=30, color=(255, 255, 255)):
    # Đường dẫn font có sẵn trên hệ thống Linux/Ubuntu của Jetson Nano
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" 
    
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        
    # Lưu ý: PIL dùng hệ màu RGB
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# --- XỬ LÝ MEDIA PIPE ---
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    wrist_global = landmarks[0].copy()
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    ref = np.linalg.norm(centered[9])
    if ref < 1e-6:
        ref = 1.0
    scaled = centered / ref
    shape_features = scaled.reshape(-1)
    combined = np.concatenate([shape_features, wrist_global])
    return combined.astype(np.float32)

def extract_frame_features(results, frame, mp_drawing, mp_hands) -> np.ndarray:
    frame_features = np.zeros(FEATURE_DIM, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            norm_pts = normalize_landmarks(pts)
            if label == "Left":
                frame_features[:66] = norm_pts
            elif label == "Right":
                frame_features[66:] = norm_pts
    return frame_features

# --- CÁC HÀM XỬ LÝ CHUỖI ---
def sample_or_pad_flat(buffer_list, K: int) -> np.ndarray:
    seq = np.array(buffer_list, dtype=np.float32)
    t = seq.shape[0]
    if t >= K:
        indices = np.linspace(0, t - 1, K, dtype=int)
        res = seq[indices]
    else:
        pad = np.zeros((K - t, seq.shape[1]), dtype=np.float32)
        res = np.vstack([seq, pad])
    return res.reshape(-1) 

def majority_vote(votes: list) -> int:
    if not votes:
        return 0
    return max(set(votes), key=votes.count)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    print("="*50)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG SIGN2VI TRÊN JETSON NANO 🚀")
    print("="*50)

    # 1. Load cấu hình
    gestures_dict = load_json("configs/gestures.json")
    sentences_dict = load_json("configs/sentences.json")
    
    gesture_id_to_name = {int(k): v for k, v in gestures_dict.items()}
    sentence_id_to_text = {int(k): v for k, v in sentences_dict.items()}

    # 2. Khởi tạo ONNX Runtime Sessions
    try:
        print("Đang load ONNX Models...")
        gesture_sess = ort.InferenceSession("models/gesture_classifier.onnx", providers=['CPUExecutionProvider'])
        seq_sess = ort.InferenceSession("models/seq_model.onnx", providers=['CPUExecutionProvider'])
        
        g_input_name = gesture_sess.get_inputs()[0].name
        s_input_name = seq_sess.get_inputs()[0].name
    except Exception as e:
        print(f"Lỗi khi load mô hình ONNX: {e}")
        print("Vui lòng đảm bảo bạn đã chạy file export_onnx.py và copy thư mục models/ sang Jetson Nano.")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Hãy kiểm tra lại kết nối!")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 3. Các Buffer xử lý luồng dữ liệu
    frame_vec_buffer = deque(maxlen=K)        
    gesture_vote_buffer = deque(maxlen=8)     
    gesture_seq_buffer = deque(maxlen=L)      
    
    frame_count = 0
    gname = "(không thấy tay)"
    sentence = "(đang nhận diện...)"

    print("✅ Hệ thống đã sẵn sàng! Mở cửa sổ Camera...")
    print("💡 Bấm 'c' để xóa chuỗi câu hiện tại (Clear).")
    print("💡 Bấm 'q' để thoát.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # --- A. PHÁT HIỆN TAY & TRÍCH XUẤT ĐẶC TRƯNG ---
            if results.multi_hand_landmarks:
                vec = extract_frame_features(results, frame, mp_drawing, mp_hands)
                
                if frame_count % 3 == 0:
                    frame_vec_buffer.append(vec)

                # --- B. NHẬN DIỆN KÝ HIỆU (Gesture Classification) ---
                if len(frame_vec_buffer) >= 6:
                    x_flat = sample_or_pad_flat(list(frame_vec_buffer), K=K)
                    x_input = np.expand_dims(x_flat, axis=0)

                    g_logits = gesture_sess.run(None, {g_input_name: x_input})[0]
                    g_probs = softmax(g_logits)
                    
                    pred_gid = np.argmax(g_probs[0])
                    conf_val = g_probs[0][pred_gid]

                    if conf_val > CONFIDENCE_THRESHOLD:
                        gesture_vote_buffer.append(pred_gid)
                    else:
                        gesture_vote_buffer.append(0) 

                    stable_gid = majority_vote(list(gesture_vote_buffer))
                    gname = gesture_id_to_name.get(stable_gid, "UNK")

                    # --- C. DỊCH CÂU (Sequence Translation) ---
                    if stable_gid != 0:
                        if len(gesture_seq_buffer) == 0 or gesture_seq_buffer[-1] != stable_gid:
                            gesture_seq_buffer.append(stable_gid)

                        if len(gesture_seq_buffer) >= 2:
                            seq = list(gesture_seq_buffer)
                            
                            if len(seq) < L:
                                seq = seq + [0] * (L - len(seq))
                            else:
                                seq = seq[-L:]
                            
                            s_input = np.array([seq], dtype=np.int64)
                            
                            s_logits = seq_sess.run(None, {s_input_name: s_input})[0]
                            sid = np.argmax(s_logits[0])
                            
                            sentence = sentence_id_to_text.get(sid, "(không xác định)")
            else:
                gname = "(không thấy tay)"

            # --- D. HIỂN THỊ KẾT QUẢ TRÊN MÀN HÌNH BẰNG PIL ---
            h, w = frame.shape[:2]
            
            # Kẻ khung đen làm nền chữ (kéo cao hơn để chứa đủ 3 dòng)
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)

            # 1. Chuyển sang RGB cho PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. Vẽ text tiếng Việt (Lưu ý: RGB thì (0, 255, 0) là xanh lá)
            frame_rgb = put_vietnamese_text(frame_rgb, f"Gesture: {gname}", (10, 10), font_size=24, color=(0, 255, 0))
            frame_rgb = put_vietnamese_text(frame_rgb, f"VI: {sentence}", (10, 45), font_size=28, color=(255, 255, 255))
            
            seq_str = " + ".join([gesture_id_to_name.get(idx, "UNK") for idx in gesture_seq_buffer])
            frame_rgb = put_vietnamese_text(frame_rgb, f"Seq: {seq_str}", (10, 85), font_size=18, color=(255, 255, 0))

            # 3. Chuyển ngược về BGR cho OpenCV
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            cv2.imshow("Sign2Vi - Realtime Inference", frame)

            # --- E. XỬ LÝ PHÍM TẮT ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                gesture_seq_buffer.clear()
                sentence = "(đã xóa chuỗi...)"
                print("Đã xóa chuỗi ký hiệu (Reset Sequence)")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()