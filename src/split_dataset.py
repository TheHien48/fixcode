import json
import random
from collections import defaultdict
from pathlib import Path

# Fix seed để kết quả chia tập luôn giống nhau trong mỗi lần chạy
SEED = 2026
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

random.seed(SEED)

root = Path("data/keypoints")
out = Path("data/splits")
out.mkdir(parents=True, exist_ok=True)

# Đọc file cấu hình để lấy danh sách các nhãn
gestures = json.load(open("configs/gestures.json", encoding="utf-8"))
label_names = sorted(gestures.values(), key=len, reverse=True)

def infer_label(path: Path) -> str:
    stem = path.stem
    for name in label_names:
        if stem == name or stem.startswith(name + "_"):
            return name
    raise ValueError(f"Không suy ra được nhãn từ: {path.name}")

by_class = defaultdict(list)

# Quét toàn bộ file .npy và phân loại theo nhãn
for path in sorted(root.glob("*.npy")):
    by_class[infer_label(path)].append(path)

splits = {"train": [], "val": [], "test": []}

# Thực hiện chia tỷ lệ 70-15-15 cho từng lớp (stratified split)
for label, files in sorted(by_class.items()):
    random.shuffle(files)
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    splits["train"].extend(files[:n_train])
    splits["val"].extend(files[n_train:n_train + n_val])
    splits["test"].extend(files[n_train + n_val:])
    
    # In báo cáo số lượng từng lớp
    print(label, "Tổng:", n, "| Train:", n_train, "| Val:", n_val, "| Test:", n - n_train - n_val)

# Ghi danh sách ra 3 file txt
for split_name, files in splits.items():
    with open(out / f"{split_name}.txt", "w", encoding="utf-8") as f:
        for path in sorted(files):
            f.write(str(path.as_posix()) + "\n")
    print(f"Đã tạo {split_name}.txt với {len(files)} file")
