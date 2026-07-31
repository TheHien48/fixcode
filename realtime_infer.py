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
CONFIDENCE_THRESHOLD = 0.6  # Ngưỡng tin cậy nhận diện ký hiệu
SENTENCE_THRESHOLD = 0.75   # Ngưỡng tin cậy để chốt câu hoàn chỉnh (Tránh AI đoán lụi)

# --- BỘ TỪ ĐIỂN DỊCH TỪNG TỪ (WORD-BY-WORD) ---
GESTURE_TO_VI = {
    "NONE": "",
    "CHAO": "Xin chào",
    "TOI": "Tôi",
    "BAN": "bạn",
    "TEN": "tên",
    "LA": "là",
    "CAM_ON": "Cảm ơn",
    "XIN_LOI": "Xin lỗi",
    "CAN": "cần",
    "GIUP": "giúp đỡ"
}

def load_json(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- HÀM VẼ TIẾNG VIỆT CÓ DẤU BẰNG PIL ---
def put_vietnamese_text(img, text, position, font_size=30, color=(255, 255, 255)):
    # Đã đổi sang file font đi kèm cùng thư mục (Mày nhớ tải Roboto.ttf hoặc arial.ttf để cùng thư mục code nhé)
    font_path = "Roboto.ttf" 
    
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback về font hệ thống của Jetson Nano nếu không tìm thấy font ở thư mục hiện tại
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
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

    gestures_dict = load_json("configs/gestures.json")
    sentences_dict = load_json("configs/sentences.json")
    
    gesture_id_to_name = {int(k): v for k, v in gestures_dict.items()}
    sentence_id_to_text = {int(k): v for k, v in sentences_dict.items()}

    try:
        print("Đang load ONNX Models...")
        gesture_sess = ort.InferenceSession("models/gesture_classifier.onnx", providers=['CPUExecutionProvider'])
        seq_sess = ort.InferenceSession("models/seq_model.onnx", providers=['CPUExecutionProvider'])
        
        g_input_name = gesture_sess.get_inputs()[0].name
        s_input_name = seq_sess.get_inputs()[0].name
    except Exception as e:
        print(f"Lỗi khi load mô hình ONNX: {e}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Hãy kiểm tra lại kết nối!")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_vec_buffer = deque(maxlen=K)        
    gesture_vote_buffer = deque(maxlen=8)     
    gesture_seq_buffer = deque(maxlen=L)      
    
    frame_count = 0
    gname = "(không có)"
    sentence = "(đang ghép câu...)"

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

            if results.multi_hand_landmarks:
                vec = extract_frame_features(results, frame, mp_drawing, mp_hands)
                
                if frame_count % 3 == 0:
                    frame_vec_buffer.append(vec)

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

                    # --- C. DỊCH CÂU & TỪNG TỪ (Sequence Translation) ---
                    if stable_gid != 0:
                        if len(gesture_seq_buffer) == 0 or gesture_seq_buffer[-1] != stable_gid:
                            gesture_seq_buffer.append(stable_gid)

                        # Bắt đầu dịch ngay từ ký hiệu đầu tiên (>=1) thay vì chờ 2 ký hiệu
                        if len(gesture_seq_buffer) >= 1:
                            seq = list(gesture_seq_buffer)
                            
                            if len(seq) < L:
                                seq = seq + [0] * (L - len(seq))
                            else:
                                seq = seq[-L:]
                            
                            s_input = np.array([seq], dtype=np.int64)
                            
                            s_logits = seq_sess.run(None, {s_input_name: s_input})[0]
                            s_probs = softmax(s_logits)[0] # Tính xác suất
                            
                            sid = np.argmax(s_probs)
                            s_conf = s_probs[sid]
                            
                            # CHỈ chốt câu hoàn chỉnh khi AI tự tin > 75%, tránh đoán bừa
                            if s_conf > SENTENCE_THRESHOLD:
                                sentence = sentence_id_to_text.get(sid, "(không xác định)")
                            else:
                                sentence = "(đang ghép câu...)"
            else:
                gname = "(không có)"

            # --- D. HIỂN THỊ KẾT QUẢ TRÊN MÀN HÌNH BẰNG PIL ---
            h, w = frame.shape[:2]
            
            # Thu nhỏ khung đen làm nền chữ (giảm từ 130px xuống 90px vì đã bỏ 1 dòng)
            cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Dòng 1: Ký hiệu gốc (TOI, TEN, LA...)
            frame_rgb = put_vietnamese_text(frame_rgb, f"Ký hiệu AI: {gname}", (10, 10), font_size=20, color=(0, 255, 0))
            
            # 2. Dòng 2: Dịch Từng Từ một theo tay người múa (Ví dụ: Tôi tên là)
            seq_words = []
            for idx in gesture_seq_buffer:
                label_en = gesture_id_to_name.get(idx, "UNK")
                vi_word = GESTURE_TO_VI.get(label_en, label_en)
                if vi_word: seq_words.append(vi_word)
            
            seq_str_vi = " ".join(seq_words)
            # Phóng to chữ dịch từng từ lên một chút cho dễ nhìn
            frame_rgb = put_vietnamese_text(frame_rgb, f"Dịch: {seq_str_vi}", (10, 45), font_size=28, color=(255, 255, 0))

            # (ĐÃ BỎ DÒNG CÂU CHỐT Ở ĐÂY)
            
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("Sign2Vi - Realtime Inference", frame)
            # 3. Dòng 3: Câu hoàn chỉnh khi đã đủ ngữ cảnh (Ví dụ: Tôi tên là ...)
            frame_rgb = put_vietnamese_text(frame_rgb, f"Câu chốt: {sentence}", (10, 85), font_size=28, color=(255, 255, 255))
            
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