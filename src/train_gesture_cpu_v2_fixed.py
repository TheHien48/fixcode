import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path
from collections import Counter


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_or_pad(seq: np.ndarray, K: int) -> np.ndarray:
    """
    Đưa chuỗi keypoints về đúng K frame.
    Input: seq (T, 132), K (số frame cố định)
    Output: np.ndarray shape (K, 132)
    """
    t = seq.shape[0]
    if t >= K:
        indices = np.linspace(0, t - 1, K, dtype=int)
        return seq[indices]
    else:
        pad = np.zeros((K - t, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])


class GestureDataset(Dataset):
    # ĐÃ SỬA: Nhận đường dẫn file txt thay vì thư mục
    def __init__(self, split_file: str, gestures_json: str, K: int = 16):
        self.K = K
        self.gestures = load_json(gestures_json)

        self.inv_gestures = {
            v.upper(): int(k)
            for k, v in self.gestures.items()
        }
        gesture_names = sorted(
            self.inv_gestures.keys(),
            key=len,
            reverse=True
        )

        self.items = []
        split_path = Path(split_file)

        if not split_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file danh sách: {split_file}")

        print(f"Đang nạp dữ liệu từ: {split_path.name}...")

        # ĐÃ SỬA: Đọc từng dòng trong file split (train.txt / val.txt)
        with open(split_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        found_counts = Counter()
        skipped_files = []

        for line in lines:
            p = Path(line)
            full_name = p.stem.upper()
            label_name = None

            for g_name in gesture_names:
                if full_name.startswith(g_name):
                    label_name = g_name
                    break

            if label_name is None:
                skipped_files.append(p)
                continue

            self.items.append((p, self.inv_gestures[label_name]))
            found_counts[label_name] += 1

        print(f"--- Thống kê {split_path.name} ---")
        for g_name, count in sorted(found_counts.items()):
            print(f"Nhãn {g_name}: {count} mẫu")
        print(f"Tổng số mẫu hợp lệ: {len(self.items)}\n")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, y = self.items[idx]
        seq = np.load(path).astype(np.float32)

        if seq.ndim != 2 or seq.shape[1] != 132:
            raise ValueError(
                f"File {path} sai shape {seq.shape}. Cần dạng (T, 132)."
            )

        x = sample_or_pad(seq, self.K)  # (K, 132)
        x = x.reshape(-1)               # (K * 132,)

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


class MLP(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cpu")
    K = 16
    feature_dim = 132
    batch_size = 32
    epochs = 100
    learning_rate = 1e-3

    # ĐÃ SỬA: Định nghĩa đường dẫn tập train và val
    train_split = "data/splits/train.txt"
    val_split = "data/splits/val.txt"
    gestures_path = "configs/gestures.json"
    save_path = "models/gesture_classifier_v1.pt"

    if not Path(gestures_path).exists():
        print(f"Lỗi: Không tìm thấy file {gestures_path}")
        return

    gestures = load_json(gestures_path)
    n_classes = len(gestures)

    print("Khởi tạo bộ dữ liệu...\n")
    try:
        train_ds = GestureDataset(split_file=train_split, gestures_json=gestures_path, K=K)
        val_ds = GestureDataset(split_file=val_split, gestures_json=gestures_path, K=K)
    except Exception as e:
        print(f"Lỗi khi load dataset: {e}")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    # Val_loader không cần shuffle (xáo trộn) vì chỉ dùng để chấm điểm
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MLP(in_dim=K * feature_dim, n_classes=n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    print("==================================================")
    print("🚀 BẮT ĐẦU HUẤN LUYỆN CHỐNG HỌC VẸT 🚀")
    print("==================================================")

    best_val_acc = 0.0 # Biến lưu đỉnh cao phong độ của AI

    for epoch in range(epochs):
        # --- BƯỚC 1: HỌC TẬP (TRAIN PHASE) ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += x.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # --- BƯỚC 2: THI THỬ (VALIDATION PHASE) ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad(): # Tắt tính toán gradient vì không học ở bước này
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits, y)

                val_loss += loss.item() * x.size(0)
                pred = logits.argmax(dim=1)
                val_correct += (pred == y).sum().item()
                val_total += x.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch + 1:03d}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} - Acc: {train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} - Acc: {val_acc:.4f}")

        # --- BƯỚC 3: LƯU PHIÊN BẢN TỐT NHẤT ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "K": K,
                    "feature_dim": feature_dim,
                    "n_classes": n_classes,
                    "gestures": gestures,
                    "state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc
                },
                save_path
            )
            print(f"  -> Đã lưu model tốt nhất (Val Acc: {best_val_acc:.4f})")

    print("\nHoàn tất huấn luyện!")
    print(f"Mô hình đạt phong độ cao nhất đã được lưu tại: {save_path} với Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()