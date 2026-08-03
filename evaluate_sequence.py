import os
import json
import numpy as np
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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

    # Lấy danh sách tên các câu theo thứ tự ID
    target_names = [sentences[str(i)] for i in range(len(sentences))]

    # 2. Load dữ liệu test chuỗi
    data_path = "data/labels/seq_samples.npz"
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy dữ liệu tại {data_path}. Hãy chạy create_seq_data.py trước!")
        return

    print("Đang nạp dữ liệu chuỗi...")
    data = np.load(data_path)
    
    # ÉP KIỂU: ONNX SeqModel yêu cầu input_type là int64 (từ torch.long)
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

    # 4. Chạy suy luận để lấy dự đoán
    print("Đang cho AI dự đoán các câu (quá trình này mất vài giây)...")
    for x in X_test:
        # Reshape từ (12,) thành (1, 12) để khớp batch_size = 1 của ONNX
        x_input = x.reshape(1, -1)
        logits = sess.run(None, {input_name: x_input})[0]
        
        # Lấy ID có xác suất cao nhất
        pred_id = np.argmax(logits[0])
        y_pred.append(pred_id)

    # 5. Tính toán và xuất báo cáo Text (Accuracy, Precision, Recall, F1)
    print("\nĐang tính toán Sentence Accuracy và F1-score...")
    report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    # In ra terminal
    print("\n" + report)

    # Lưu vĩnh viễn vào file txt
    with open("bao_cao_seq_jetson.txt", "w", encoding="utf-8") as f:
        f.write("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH DỊCH CÂU (GRU) TRÊN JETSON NANO\n")
        f.write("="*65 + "\n")
        f.write(report)
    print("✅ Đã lưu kết quả thành công vào file 'bao_cao_seq_jetson.txt'!")

    # 6. Vẽ và lưu Ma trận nhầm lẫn (Confusion Matrix)
    print("Đang vẽ Ma trận nhầm lẫn (Confusion Matrix)...")
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    
    # Xoay nhãn trục X một góc 45 độ vì tên câu khá dài, tránh bị đè lên nhau
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45, values_format="d")
    
    plt.tight_layout()
    plt.savefig("sentence_confusion_matrix.png", dpi=300)
    print("✅ Đã lưu hình ảnh trực quan vào 'sentence_confusion_matrix.png'!")

if __name__ == "__main__":
    main()
