import time
import numpy as np
import onnxruntime as ort

def run_benchmark(session, input_name, dummy_input, num_warmup=100, num_measure=1000, model_name=""):
    print(f"\n--- Đang đo hiệu năng: {model_name} ---")
    
    # 1. Warm-up: Chạy không ghi nhận thời gian để GPU/CPU ổn định
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy_input})
        
    # 2. Đo lường chính thức
    latencies = []
    for _ in range(num_measure):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0) # Lưu dưới dạng milliseconds (ms)
        
    # 3. Tính toán thống kê
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0
    
    print(f"Số mẫu đo nghiệm      : {num_measure} frames")
    print(f"Tốc độ khung hình (FPS): {fps:.2f} FPS")
    print(f"Độ trễ trung bình     : {mean_lat:.2f} ms (± {std_lat:.2f} ms)")
    print(f"Độ trễ P50 (Median)   : {p50_lat:.2f} ms")
    print(f"Độ trễ P95            : {p95_lat:.2f} ms")
    
    return fps, mean_lat, p95_lat

def main():
    print("="*50)
    print("🚀 BẮT ĐẦU BENCHMARK HỆ THỐNG TRÊN JETSON NANO 🚀")
    print("="*50)
    
    # Cấu hình kiến trúc giống lúc Train
    K = 16
    FEATURE_DIM = 132 # Kích thước vector đặc trưng 
    L = 12            # Độ dài chuỗi từ đơn
    
    # Load Model 1: Nhận diện Ký hiệu (Gesture)
    try:
        g_sess = ort.InferenceSession("models/gesture_classifier.onnx", providers=['CPUExecutionProvider'])
        g_input_name = g_sess.get_inputs()[0].name
        # Dummy data cho MLP (1, K * FEATURE_DIM)
        g_dummy = np.random.randn(1, K * FEATURE_DIM).astype(np.float32)
        run_benchmark(g_sess, g_input_name, g_dummy, model_name="Gesture Classifier (MLP)")
    except Exception as e:
        print(f"Lỗi benchmark Gesture Model: {e}")

    # Load Model 2: Dịch câu (SeqModel)
    try:
        s_sess = ort.InferenceSession("models/seq_model.onnx", providers=['CPUExecutionProvider'])
        s_input_name = s_sess.get_inputs()[0].name
        # Dummy data cho GRU (1, L) kiểu long/int64
        s_dummy = np.zeros((1, L), dtype=np.int64)
        run_benchmark(s_sess, s_input_name, s_dummy, model_name="Sequence Model (GRU)")
    except Exception as e:
        print(f"Lỗi benchmark SeqModel: {e}")

if __name__ == "__main__":
    main()