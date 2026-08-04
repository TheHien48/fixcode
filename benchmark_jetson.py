import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import json
import time
import csv
import argparse
from collections import deque
from pathlib import Path
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# --- CẤU HÌNH CƠ BẢN ---
K = 16
FEATURE_DIM = 132
L = 12
CONFIDENCE_THRESHOLD = 0.6

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
    if not votes: return 0
    return max(set(votes), key=votes.count)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def main():
    parser = argparse.ArgumentParser(description="Sign2Vi Jetson Benchmark")
    parser.add_argument("--backend", type=str, default="onnx", choices=["onnx", "tensorrt"], help="Backend suy luận")
    parser.add_argument("--frames", type=int, default=1000, help="Số frame cần đo")
    parser.add_argument("--warmup", type=int, default=100, help="Số frame bỏ qua ban đầu")
    args = parser.parse_args()

    print("="*50)
    print(f"🚀 KHỞI ĐỘNG HỆ THỐNG BENCHMARK (Backend: {args.backend.upper()}) 🚀")
    print("="*50)

    # Khai báo provider chạy mô hình
    if args.backend == "tensorrt":
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider'] 

    try:
        gesture_session = ort.InferenceSession("models/gesture_classifier.onnx", providers=providers)
        seq_session = ort.InferenceSession("models/seq_model.onnx", providers=providers)
        g_input_name = gesture_session.get_inputs()[0].name
        s_input_name = seq_session.get_inputs()[0].name
    except Exception as e:
        print(f"Lỗi khi load mô hình ONNX: {e}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera.")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_vec_buffer = deque(maxlen=K)        
    gesture_vote_buffer = deque(maxlen=8)     
    gesture_seq_buffer = deque(maxlen=L)      
    
    records = []
    measured = 0
    frame_index = 0
    measure_start = None

    print(f"Đang Warm-up {args.warmup} frames đầu tiên...")

    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.7,
    ) as hands:
        while measured < args.frames:
            t_loop0 = time.perf_counter()
            
            t0 = time.perf_counter()
            ret, frame = cap.read()
            t1 = time.perf_counter()
            
            if not ret: 
                continue

            # MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            t2 = time.perf_counter()
            
            gesture_ms = 0.0
            sequence_ms = 0.0

            if results.multi_hand_landmarks:
                vec = extract_frame_features(results, frame, mp_drawing, mp_hands)
                if frame_index % 3 == 0:
                    frame_vec_buffer.append(vec)

                if len(frame_vec_buffer) >= 6:
                    x_flat = sample_or_pad_flat(list(frame_vec_buffer), K=K)
                    x_input = np.expand_dims(x_flat, axis=0)

                    # Model 1: Gesture Classifier
                    tg0 = time.perf_counter()
                    g_logits = gesture_session.run(None, {g_input_name: x_input})[0]
                    tg1 = time.perf_counter()
                    gesture_ms = (tg1 - tg0) * 1000.0

                    g_probs = softmax(g_logits)
                    pred_gid = np.argmax(g_probs[0])
                    conf_val = g_probs[0][pred_gid]

                    if conf_val > CONFIDENCE_THRESHOLD:
                        gesture_vote_buffer.append(pred_gid)
                    else:
                        gesture_vote_buffer.append(0) 

                    stable_gid = majority_vote(list(gesture_vote_buffer))

                    if stable_gid != 0:
                        if len(gesture_seq_buffer) == 0 or gesture_seq_buffer[-1] != stable_gid:
                            gesture_seq_buffer.append(stable_gid)
                            
                            # Model 2: Sequence Translator
                            seq = list(gesture_seq_buffer)
                            if len(seq) < L: seq = seq + [0] * (L - len(seq))
                            else: seq = seq[-L:]
                            
                            s_input = np.array([seq], dtype=np.int64)
                            
                            ts0 = time.perf_counter()
                            s_logits = seq_session.run(None, {s_input_name: s_input})[0]
                            ts1 = time.perf_counter()
                            sequence_ms = (ts1 - ts0) * 1000.0

            # KHÔNG gọi cv2.imshow() ở đây theo đúng nguyên tắc benchmark (tránh nhiễu latency UI)
            t3 = time.perf_counter()
            
            frame_index += 1
            if frame_index <= args.warmup:
                continue
                
            if measure_start is None:
                measure_start = time.perf_counter()
                print(f"Bắt đầu đo {args.frames} frames...")

            total_ms = (t3 - t_loop0) * 1000.0
            records.append({
                "frame": measured,
                "capture_ms": (t1 - t0) * 1000.0,
                "mediapipe_ms": (t2 - t1) * 1000.0,
                "gesture_ms": gesture_ms,
                "sequence_ms": sequence_ms,
                "total_ms": total_ms,
            })
            measured += 1
            
            if measured % 100 == 0:
                print(f"Đã đo: {measured}/{args.frames} frames")

    elapsed = time.perf_counter() - measure_start
    fps = measured / elapsed
    lat = np.array([r["total_ms"] for r in records])
    
    summary = {
        "backend": args.backend,
        "frames": measured,
        "elapsed_s": elapsed,
        "fps": fps,
        "latency_mean_ms": float(np.mean(lat)),
        "latency_std_ms": float(np.std(lat, ddof=1)),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_max_ms": float(np.max(lat)),
    }

    Path("results").mkdir(exist_ok=True)
    csv_file_path = f"results/jetson_frame_timing_{args.backend}.csv"
    with open(csv_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    summary_file_path = f"results/jetson_benchmark_summary_{args.backend}.csv"
    with open(summary_file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)

    print("\n" + "="*50)
    print("TỔNG KẾT BENCHMARK")
    print("="*50)
    print(f"Backend           : {args.backend.upper()}")
    print(f"Độ phân giải      : 640x480")
    print(f"Tổng số frame     : {summary['frames']}")
    print(f"FPS               : {summary['fps']:.2f}")
    print(f"Latency Mean (ms) : {summary['latency_mean_ms']:.2f} ± {summary['latency_std_ms']:.2f}")
    print(f"Latency P50 (ms)  : {summary['latency_p50_ms']:.2f}")
    print(f"Latency P95 (ms)  : {summary['latency_p95_ms']:.2f}")
    print(f"Chi tiết lưu tại  : {summary_file_path}")
    print("="*50)

if __name__ == '__main__':
    main()
#python3 src/benchmark_jetson.py --backend onnx --frames 1000 --warmup 100