import os
import json
import glob
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report

# Cấu hình K giống như lúc train
K = 16

# 1. Hàm tiền xử lý giống hệt file train
def sample_or_pad(seq, K):
    t = seq.shape[0]
    if t >= K:
        indices = np.linspace(0, t - 1, K, dtype=int)
        return seq[indices]
    else:
        pad = np.zeros((K - t, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])

def main():
    print("="*60)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH TRÊN JETSON NANO 🚀")
    print("="*60)
    
    # 2. Load danh sách ký hiệu
    try:
        with open("configs/gestures.json", "r", encoding="utf-8") as f:
            gestures = json.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file configs/gestures.json")
        return

    inv_gestures = {v.upper(): int(k) for k, v in gestures.items()}
    gesture_names = sorted(inv_gestures.keys(), key=len, reverse=True)
    target_names = [gestures[str(i)] for i in range(len(gestures))]

    # 3. Load model ONNX siêu nhẹ
    print("Đang nạp mô hình ONNX...")
    sess = ort.InferenceSession("models/gesture_classifier.onnx", providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    y_true = []
    y_pred = []

    # 4. Quét thư mục chứa dữ liệu
    # Ưu tiên chỉ test trên tập test.txt để kết quả khách quan nhất
    test_file = "data/splits/test.txt"
    if os.path.exists(test_file):
        with open(test_file, "r") as f:
            npy_files = [line.strip() for line in f if line.strip()]
        print(f"Đã nạp {len(npy_files)} file từ danh sách tập Test.")
    else:
        # Nếu chưa chia tập, quét toàn bộ
        npy_files = glob.glob("data/keypoints/*.npy")
        print(f"Không tìm thấy file chia tập, đang quét toàn bộ {len(npy_files)} file...")

    if len(npy_files) == 0:
        print("Lỗi: Không có dữ liệu để đánh giá!")
        return

    print("Đang cho AI dự đoán (quá trình này mất vài giây)...")
    for filepath in npy_files:
        filename = os.path.basename(filepath).upper()
        label_name = None
        
        # Lấy nhãn thật từ tên file
        for g_name in gesture_names:
            if filename.startswith(g_name):
                label_name = g_name
                break
        
        if label_name is None:
            continue
            
        true_id = inv_gestures[label_name]
        
        # Đọc dữ liệu (T, 132) và xử lý
        seq = np.load(filepath).astype(np.float32)
        x = sample_or_pad(seq, K)
        x_flat = x.reshape(1, -1) # Thêm chiều batch_size = 1 cho ONNX
        
        # Chạy ONNX dự đoán
        logits = sess.run(None, {input_name: x_flat})[0]
        pred_id = np.argmax(logits[0])
        
        y_true.append(true_id)
        y_pred.append(pred_id)

    # 5. Tính toán và xuất báo cáo
    print("\nĐang tính toán Accuracy, Precision, Recall, F1-score...")
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    # In ra terminal
    print("\n" + report)

    # Lưu vĩnh viễn vào file txt
    with open("bao_cao_jetson.txt", "w", encoding="utf-8") as f:
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH NHẬN DIỆN KÝ HIỆU TRÊN JETSON NANO\n")
        f.write("="*65 + "\n")
        f.write(report)
    print("✅ Đã lưu kết quả thành công vào file 'bao_cao_jetson.txt'!")

if __name__ == "__main__":
    main()