import os
# =========================================================================
# BÙA CHÚA TỂ: KHÓA MÕM LÕI PROTOBUF C++ (THỦ PHẠM GÂY SEGFAULT 100%)
# =========================================================================
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
# Ép xài OpenCV xịn của NVIDIA
if '/usr/lib/python3.6/dist-packages' not in sys.path:
    sys.path.insert(1, '/usr/lib/python3.6/dist-packages')

import io
import time
import json
import numpy as np
from collections import deque
from pathlib import Path

# Ép terminal in tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60, flush=True)
print("🚀 HỆ THỐNG SIGN2VI - FIX TRIỆT ĐỂ LỖI PROTOBUF 🚀", flush=True)
print("="*60, flush=True)

# 1. Kích hoạt MediaPipe đầu tiên
print("⏳ [1/3] Kích hoạt MediaPipe (Đã tiêm thuốc giải Protobuf)...", flush=True)
import mediapipe as mp
hands = mp.solutions.hands.Hands(
    static_image_mode=True, 
    max_num_hands=2,
    min_detection_confidence=0.7
)
print("✅ MediaPipe an toàn tuyệt đối!", flush=True)

# 2. Nạp OpenCV xịn và ONNX
print("⏳ [2/3] Nạp OpenCV xịn và AI ONNX...", flush=True)
import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
import onnxruntime as ort
print("✅ Thư viện sẵn sàng!", flush=True)

# --- CẤU HÌNH CƠ BẢN ---
K = 16
FEATURE_DIM = 132
L = 12
CONFIDENCE_THRESHOLD = 0.6  
SENTENCE_THRESHOLD = 0.75   
COOLDOWN_TIME = 2.5         

GESTURE_TO_VI = {
    "NONE": "", "CHAO": "Xin chào", "TOI": "Tôi", "BAN": "bạn",
    "TEN": "tên", "LA": "là", "CAM_ON": "Cảm ơn", "XIN_LOI": "Xin lỗi",
    "CAN": "cần", "GIUP": "giúp đỡ"
}

def load_json(path: str) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Lỗi: Không tìm thấy file {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    wrist_global = landmarks[0].copy()
    wrist = landmarks[0].copy()
    centered = landmarks - wrist
    ref = np.linalg.norm(centered[9])
    if ref < 1e-6: ref = 1.0
    scaled = centered / ref
    shape_features = scaled.reshape(-1)
    return np.concatenate([shape_features, wrist_global]).astype(np.float32)

def extract_frame_features(results) -> np.ndarray:
    frame_features = np.zeros(FEATURE_DIM, dtype=np.float32)
    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=np.float32)
            norm_pts = normalize_landmarks(pts)
            if label == "Left": frame_features[:66] = norm_pts
            elif label == "Right": frame_features[66:] = norm_pts
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
    if not votes: return 0
    return max(set(votes), key=votes.count)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    gestures_dict = load_json("configs/gestures.json")
    sentences_dict = load_json("configs/sentences.json")
    gesture_id_to_name = {int(k): v for k, v in gestures_dict.items()}
    sentence_id_to_text = {int(k): v for k, v in sentences_dict.items()}
    
    print("⏳ [3/3] Bật AI ONNX và kết nối Camera...", flush=True)
    try:
        sess_opt = ort.SessionOptions()
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        gesture_sess = ort.InferenceSession("models/gesture_classifier.onnx", sess_options=sess_opt, providers=['CPUExecutionProvider'])
        seq_sess = ort.InferenceSession("models/seq_model.onnx", sess_options=sess_opt, providers=['CPUExecutionProvider'])
        
        g_input_name = gesture_sess.get_inputs()[0].name
        s_input_name = seq_sess.get_inputs()[0].name
    except Exception as e:
        print(f"❌ Lỗi khi load ONNX: {e}", flush=True)
        return

    frame_vec_buffer = deque(maxlen=K)        
    gesture_vote_buffer = deque(maxlen=8)     
    gesture_seq_buffer = deque(maxlen=L)      
    
    frame_count = 0
    lock_until = 0.0
    
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ LỖI: Không mở được Camera!", flush=True)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n💡 HỆ THỐNG ĐÃ LÊN SÓNG! Múa ngay đi người anh em!", flush=True)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            current_time = time.time()
            frame_count += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                vec = extract_frame_features(results)
                if frame_count % 3 == 0: frame_vec_buffer.append(vec)

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
                                
                                print(f"🎯 KẾT QUẢ CHỐT CÂU: {final_sentence} (Từ: {gname})", flush=True)
                                gesture_seq_buffer.clear()
            else:
                if frame_count % 60 == 0:
                    print("⏳ Đang quét camera... (Chưa thấy tay)", flush=True)

    except KeyboardInterrupt:
        print("\n🛑 Đã dừng chương trình an toàn.", flush=True)
    finally:
        hands.close()
        cap.release()

if __name__ == "__main__":
    main()