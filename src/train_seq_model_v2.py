import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path

# --- ĐÃ HOÀN THIỆN: Định nghĩa bộ nạp dữ liệu chuỗi câu ---
class SeqDataset(Dataset):
    def __init__(self, npz_path: str):
        # Load file dữ liệu chuỗi câu (.npz)
        obj = np.load(npz_path)
        self.X = obj["X"]  # Mảng chứa chuỗi ID của các từ đơn, kích thước (N, Độ dài chuỗi)
        self.y = obj["y"]  # Mảng chứa ID của câu hoàn chỉnh tương ứng, kích thước (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Trả về chuỗi ID từ đơn và ID câu tương ứng dưới dạng Tensor Long
        return (
            torch.tensor(self.X[idx], dtype=torch.long), 
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# --- Mạng GRU xử lý chuỗi ngữ cảnh ---
class SeqModel(nn.Module):
    def __init__(self, vocab_size: int, n_sentences: int, emb_dim: int = 32, hid: int = 64):
        super().__init__()
        # Tầng nhúng: Biến ID từ đơn thành Vector 32 chiều
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # Tầng GRU: Đọc chuỗi từ để hiểu ngữ cảnh câu
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hid, batch_first=True, num_layers=1)
        # Tầng tuyến tính: Phân loại ra ID của câu dịch hoàn chỉnh
        self.fc = nn.Linear(hid, n_sentences)

    def forward(self, x):
        e = self.emb(x)
        out, _ = self.rnn(e)
        last = out[:, -1, :]  # Bốc lấy đặc trưng của từ cuối cùng trong chuỗi
        return self.fc(last)


def main():
    device = torch.device("cpu")

    # Load file cấu hình cấu trúc hệ thống
    gestures = json.load(open("configs/gestures.json", "r", encoding="utf-8"))
    sentences = json.load(open("configs/sentences.json", "r", encoding="utf-8"))

    vocab_size = len(gestures)      # Tổng số từ đơn hệ thống hỗ trợ
    n_sentences = len(sentences)    # Tổng số câu dịch đầu ra

    npz_path = "data/labels/seq_samples.npz"
    if not Path(npz_path).exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu mẫu tại {npz_path}")
        return

    ds = SeqDataset(npz_path)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    model = SeqModel(vocab_size=vocab_size, n_sentences=n_sentences).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("Bắt đầu huấn luyện Tầng Dịch Chuỗi Ngôn Ngữ (SeqModel)...")
    for epoch in range(25):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        print(f"Epoch {epoch+1:02d} | loss={loss_sum/total:.4f} | acc={correct/total:.4f}")

    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/seq_model_cpu.pt")
    print("\nĐã lưu mô hình dịch ngôn ngữ thành công tại: models/seq_model_cpu.pt")


if __name__ == "__main__":
    main()