import json
import torch
import torch.nn as nn
from pathlib import Path

# --- ĐÃ SỬA: Thêm Dropout cho khớp 100% với kiến trúc lúc Train ---
class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2), # Khớp với file Train
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Khớp với file Train
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)


class SeqModel(nn.Module):
    def __init__(self, vocab_size: int, n_sentences: int, emb_dim: int = 32, hid: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hid, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hid, n_sentences)
        
    def forward(self, x):
        e = self.emb(x)
        out, _ = self.rnn(e)
        last = out[:, -1, :]
        return self.fc(last)


def export_gesture():
    # ĐÃ SỬA: Gọi đúng tên file v1.pt đã train
    model_path = "models/gesture_classifier_v1.pt" 
    if not Path(model_path).exists():
        print(f"Lỗi: Không tìm thấy file {model_path}. Hãy chạy train trước!")
        return

    ckpt = torch.load(model_path, map_location="cpu")
    K = ckpt["K"]
    # ĐÃ SỬA: Lấy động feature_dim từ checkpoint (sẽ ra 132 thay vì hardcode 63)
    feature_dim = ckpt.get("feature_dim", 132) 
    n_classes = ckpt["n_classes"]
    
    # Khởi tạo mô hình chuẩn số chiều mới (K * 132)
    model = MLP(in_dim=K * feature_dim, n_classes=n_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Tạo dữ liệu giả lập chuẩn shape (1, 2112) để làm bản vẽ xuất ONNX
    dummy = torch.randn(1, K * feature_dim)
    
    out_onnx = "models/gesture_classifier.onnx"
    torch.onnx.export(
        model, 
        dummy,
        out_onnx,
        input_names=["x"],
        output_names=["logits"],
        opset_version=13
    )
    print(f"Đã xuất Gesture Model ONNX thành công: {out_onnx}")


def export_seq():
    model_path = "models/seq_model_cpu.pt"
    if not Path(model_path).exists():
        print(f"Lỗi: Không tìm thấy file {model_path}. Hãy chạy train mẫu câu trước!")
        return

    gestures = json.load(open("configs/gestures.json", "r", encoding="utf-8"))
    sentences = json.load(open("configs/sentences.json", "r", encoding="utf-8"))

    model = SeqModel(vocab_size=len(gestures), n_sentences=len(sentences))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Giữ nguyên mảng long (1, 12) giả lập chuỗi câu đầu vào gồm 12 từ đơn
    dummy = torch.zeros((1, 12), dtype=torch.long)
    
    out_onnx = "models/seq_model.onnx"
    torch.onnx.export(
        model, 
        dummy,
        out_onnx,
        input_names=["x"],
        output_names=["logits"],
        opset_version=11
    )
    print(f"Đã xuất Seq Model ONNX thành công: {out_onnx}")


if __name__ == "__main__":
    export_gesture()
    export_seq()
    print("\n[DONE] Toàn bộ hệ thống đã được đóng gói ONNX tại thư mục models/")