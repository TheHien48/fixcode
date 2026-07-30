import os
# Cấm đa luồng từ mức hệ thống để chống tràn RAM/CPU trên Jetson Nano
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 1. BẮT BUỘC IMPORT OPENCV ĐẦU TIÊN ĐỂ XÍ CHỖ ĐỒ HỌA
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)

# 2. IMPORT MEDIAPIPE THỨ HAI
import mediapipe as mp

# 3. IMPORT ONNXRUNTIME CUỐI CÙNG
import onnxruntime as ort

import numpy as np
import json
import time
from collections import deque
from pathlib import Path
import sys
import io

# --- Ép terminal dùng UTF-8 để in log tiếng Việt không lỗi ---
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CẤU HÌNH CƠ BẢN ---
K = 16
FEATURE_DIM = 132
L = 12
CONFIDENCE_THRESHOLD = 0.6  
SENTENCE_THRESHOLD = 0.75   
COOLDOWN_TIME = 2.5         

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

def extract_frame_features(results, frame, mp_hands) -> np.ndarray:
    frame_features = np.zeros(FEATURE_DIM, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            norm_pts = normalize_landmarks(pts)
            if label == "Left":
                frame_features[:66] = norm_pts
            elif label == "Right":
                frame_features[66:] = norm_pts
    return frame_features

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
    print("="*60)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG SIGN2VI - CHẾ ĐỘ CHẠY NGẦM 🚀")
    print("="*60)

    gestures_dict = load_json("configs/gestures.json")
    sentences_dict = load_json("configs/sentences.json")
    
    gesture_id_to_name = {int(k): v for k, v in gestures_dict.items()}
    sentence_id_to_text = {int(k): v for k, v in sentences_dict.items()}
    
    # 1. KHỞI TẠO CAMERA TRƯỚC (Dùng V4L2 cho Camera USB)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Hãy kiểm tra lại kết nối!")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. KHỞI TẠO ONNX
    try:
        print("Đang load ONNX Models...")
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        gesture_sess = ort.InferenceSession("models/gesture_classifier.onnx", sess_options=sess_opt, providers=['CPUExecutionProvider'])
        seq_sess = ort.InferenceSession("models/seq_model.onnx", sess_options=sess_opt, providers=['CPUExecutionProvider'])
        
        g_input_name = gesture_sess.get_inputs()[0].name
        s_input_name = seq_sess.get_inputs()[0].name
    except Exception as e:
        print(f"Lỗi khi load mô hình ONNX: {e}")
        return

    mp_hands = mp.solutions.hands

    frame_vec_buffer = deque(maxlen=K)        
    gesture_vote_buffer = deque(maxlen=8)     
    gesture_seq_buffer = deque(maxlen=L)      
    
    frame_count = 0
    gname = "(không có)"
    lock_until = 0.0

    print("✅ Hệ thống đã sẵn sàng! Bắt đầu thu nhận hình ảnh...")
    print("💡 Đang chạy chế độ NGẦM (Không hiển thị cửa sổ để chống sập đồ họa EGL).")
    print("💡 Bấm Ctrl + C trong Terminal để dừng chương trình bất cứ lúc nào.\n")
    
    # 3. KHỞI TẠO MEDIAPIPE (Đã bỏ model_complexity)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                current_time = time.time()
                frame_count += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    vec = extract_frame_features(results, frame, mp_hands)
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

                        if current_time > lock_until:
                            if stable_gid != 0:
                                if len(gesture_seq_buffer) == 0 or gesture_seq_buffer[-1] != stable_gid:
                                    gesture_seq_buffer.append(stable_gid)
                                    
                            if len(gesture_seq_buffer) >= 1:
                                seq = list(gesture_seq_buffer)
                                if len(seq) < L:
                                    seq = seq + [0] * (L - len(seq))
                                else:
                                    seq = seq[-L:]
                                
                                s_input = np.array([seq], dtype=np.int64)
                                s_logits = seq_sess.run(None, {s_input_name: s_input})[0]
                                s_probs = softmax(s_logits)[0]
                                
                                sid = np.argmax(s_probs)
                                s_conf = s_probs[sid]
                                
                                if s_conf > SENTENCE_THRESHOLD and sid != 0:
                                    final_sentence = sentence_id_to_text.get(sid, "(không xác định)")
                                    lock_until = current_time + COOLDOWN_TIME
                                    
                                    # IN KẾT QUẢ RA TERMINAL 
                                    print(f"🎯 KẾT QUẢ CHỐT CÂU: {final_sentence} (Từ đang nhận: {gname})")
                                    gesture_seq_buffer.clear()
                        else:
                            pass 
                else:
                    # Báo hiệu máy vẫn đang chạy nếu không thấy tay
                    if frame_count % 60 == 0:
                        print("⏳ Đang quét camera... (Chưa thấy tay)")

        except KeyboardInterrupt:
            print("\n🛑 Đã dừng chương trình an toàn.")

    cap.release()

if __name__ == "__main__":
    main()