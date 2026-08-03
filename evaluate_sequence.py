import os
import json
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("="*65)
    print("🚀 ĐÁNH GIÁ MÔ HÌNH DỊCH CÂU (SEQ MODEL) TRÊN JETSON NANO 🚀")
    print("="*65)

    # 1. Load config sentences.json để lấy nhãn câu
    try:
        with open("configs/sentences.json", "r", encoding="utf-8") as f:
            sentences = json.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy configs/sentences.json")
        return

    target_names = [sentences[str(i)] for i in range(len(sentences))]

    # 2. Load dữ liệu test chuỗi
    data_path = "data/labels/seq_samples.npz"
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy dữ liệu tại {data_path}")
        return

    print("Đang nạp dữ liệu chuỗi...")
    data = np.load(data_path)
    X_test = data['X'].astype(np.int64) 
    y_true = data['y']
    
    print(f"Tổng số mẫu test: {len(y_true)} chuỗi")

    # 3. Load ONNX Model
    print("Đang nạp mô hình seq_model.onnx...")
    try:
        sess = ort.InferenceSession("models/seq_model.onnx", providers=['CPUExecutionProvider'])
        input_name = sess.get_inputs()[0].name
    except Exception as e:
        print(f"Lỗi khi nạp mô hình ONNX: {e}")
        return

    y_pred = []

    # 4. Chạy suy luận
    print("Đang cho AI dự đoán các câu (quá trình này mất vài giây)...")
    for x in X_test:
        x_input = x.reshape(1, -1)
        logits = sess.run(None, {input_name: x_input})[0]
        y_pred.append(np.argmax(logits[0]))

    # 5. Tính toán và xuất báo cáo Text (Accuracy, Precision, Recall, F1)
    print("\nĐang tính toán Sentence Accuracy và F1-score...")
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    print("\n" + report)

    with open("bao_cao_seq_jetson.txt", "w", encoding="utf-8") as f:
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH DỊCH CÂU (GRU) TRÊN JETSON NANO\n")
        f.write("="*65 + "\n")
        f.write(report)
    print("✅ Đã lưu kết quả Text vào 'bao_cao_seq_jetson.txt'!")

    # 6. Tính và lưu Ma trận nhầm lẫn (Confusion Matrix) ra file CSV
    print("\nĐang xuất Ma trận nhầm lẫn ra file CSV...")
    cm = confusion_matrix(y_true, y_pred)
    
    # In ra terminal cho dễ nhìn
    print(cm)
    
    # Ghi ra file CSV
    csv_path = "sentence_confusion_matrix.csv"
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        # Ghi Header
        f.write("Thực Tế \\ Dự Đoán," + ",".join(target_names) + "\n")
        # Ghi từng hàng dữ liệu
        for i, row in enumerate(cm):
            f.write(target_names[i] + "," + ",".join(map(str, row)) + "\n")
            
    print(f"✅ Đã lưu ma trận nhầm lẫn dạng bảng vào '{csv_path}'!")
    print("💡 Mẹo: Chép file CSV này sang máy tính để chèn vào báo cáo hoặc vẽ biểu đồ!")

if __name__ == "__main__":
    main()