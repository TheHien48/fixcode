import json
import numpy as np
from pathlib import Path

def main():
    # 1. Đảm bảo thư mục tồn tại
    Path("data/labels").mkdir(parents=True, exist_ok=True)

    # 2. Đọc file gestures.json để lấy ID của từng từ
    gestures_path = "configs/gestures.json"
    if not Path(gestures_path).exists():
        print(f"Lỗi: Không tìm thấy {gestures_path}.")
        return
        
    with open(gestures_path, "r", encoding="utf-8") as f:
        gestures = json.load(f)
    
    # Tạo từ điển ngược để tra ID bằng chữ (Ví dụ: "TOI" -> 2)
    inv_g = {v.upper(): int(k) for k, v in gestures.items()}

    # 3. ĐỊNH NGHĨA LUẬT GHÉP CÂU (Dựa trên 6 câu mày vừa gửi)
    X_data = []
    y_data = []

    # Tạo 500 mẫu cho mỗi câu để AI (GRU) có đủ data học ngữ cảnh
    for _ in range(500):
        # ID 0: "(không xác định)" -> Ký hiệu: NONE
        X_data.append([inv_g.get("NONE", 0)])
        y_data.append(0)

        # ID 1: "Xin chào!" -> Ký hiệu: CHAO
        X_data.append([inv_g.get("CHAO", 1)])
        y_data.append(1)

        # ID 2: "Tôi tên là ..." -> Ký hiệu: TOI + TEN + LA
        X_data.append([inv_g.get("TOI", 2), inv_g.get("TEN", 4), inv_g.get("LA", 5)])
        y_data.append(2)

        # ID 3: "Cảm ơn bạn." -> Ký hiệu: CAM_ON + BAN (hoặc chỉ CAM_ON)
        X_data.append([inv_g.get("CAM_ON", 6), inv_g.get("BAN", 3)])
        y_data.append(3)

        # ID 4: "Xin lỗi." -> Ký hiệu: XIN_LOI
        X_data.append([inv_g.get("XIN_LOI", 7)])
        y_data.append(4)

        # ID 5: "Tôi cần giúp đỡ." -> Ký hiệu: TOI + CAN + GIUP
        X_data.append([inv_g.get("TOI", 2), inv_g.get("CAN", 8), inv_g.get("GIUP", 9)])
        y_data.append(5)

    # 4. PADDING (Chuẩn hóa độ dài mảng thành L=12 như quy định của mô hình GRU)
    MAX_LEN = 12
    X_padded = []
    for seq in X_data:
        if len(seq) < MAX_LEN:
            # Nhét thêm số 0 vào cuối cho đủ 12 slot
            padded_seq = seq + [0] * (MAX_LEN - len(seq))
        else:
            padded_seq = seq[:MAX_LEN]
        X_padded.append(padded_seq)

    # Chuyển đổi sang định dạng Tensor của Numpy
    X_arr = np.array(X_padded, dtype=np.int32)
    y_arr = np.array(y_data, dtype=np.int32)

    # 5. XUẤT FILE .NPZ
    npz_path = "data/labels/seq_samples.npz"
    np.savez(npz_path, X=X_arr, y=y_arr)
    
    print(f"Đã tạo thành công file dữ liệu chuỗi tại: {npz_path}")
    print(f"Kích thước X (Dữ liệu đầu vào): {X_arr.shape}")
    print(f"Kích thước y (Nhãn câu): {y_arr.shape}")

if __name__ == "__main__":
    main()