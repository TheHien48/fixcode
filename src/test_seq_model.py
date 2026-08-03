import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path

# Cấu trúc mạng GRU (Phải copy lại giống y hệt file train để load trọng số)
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

def main():
    device = torch.device("cpu")
    
    # 1. Load từ điển và câu
    try:
        with open("configs/gestures.json", "r", encoding="utf-8") as f:
            gestures = json.load(f)
        with open("configs/sentences.json", "r", encoding="utf-8") as f:
            sentences = json.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file JSON trong thư mục configs/")
        return

    inv_g = {v.upper(): int(k) for k, v in gestures.items()}

    # 2. Load Model đã train
    model_path = "models/seq_model_cpu.pt"
    if not Path(model_path).exists():
        print(f"Lỗi: Chưa có file mô hình {model_path}. Hãy chạy train_seq_model_v2.py trước.")
        return

    model = SeqModel(vocab_size=len(gestures), n_sentences=len(sentences)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Chuyển sang chế độ test

    print("\n" + "="*50)
    print("🚀 BỘ CÔNG CỤ TEST AI DỊCH CÂU (NLP LAYER) 🚀")
    print("="*50)
    print("Nhập các từ ký hiệu viết hoa cách nhau bằng khoảng trắng.")
    print("Ví dụ: TOI TEN LA")
    print("Gõ 'Q' để thoát.\n")

    MAX_LEN = 12

    # 3. Vòng lặp Test Terminal
    while True:
        text = input(">> Nhập chuỗi ký hiệu: ").strip().upper()
        if text == 'Q':
            print("Đã thoát công cụ test.")
            break
        if not text:
            continue

        words = text.split()
        seq_ids = []
        valid = True
        
        # Biến chữ thành ID
        for w in words:
            if w not in inv_g:
                print(f"  [!] Lỗi: Từ '{w}' không có trong từ điển gestures.json!")
                valid = False
                break
            seq_ids.append(inv_g[w])

        if not valid:
            continue

        # Padding (Bù số 0 cho đủ 12 slot)
        if len(seq_ids) < MAX_LEN:
            padded_seq = seq_ids + [0] * (MAX_LEN - len(seq_ids))
        else:
            padded_seq = seq_ids[:MAX_LEN]

        # Chuyển thành Tensor và ném vào model
        x_input = torch.tensor([padded_seq], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x_input)
            probs = torch.softmax(logits, dim=1)
            conf, pred_id = torch.max(probs, dim=1)

            sid = str(pred_id.item())
            confidence = conf.item() * 100
            sentence_text = sentences.get(sid, "(Không xác định)")

        # In kết quả
        print(f"  => CÂU DỊCH: {sentence_text}")
        print(f"  => ĐỘ TIN CẬY: {confidence:.2f}%\n")

if __name__ == "__main__":
    main()