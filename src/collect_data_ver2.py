import time
import cv2
import json
from pathlib import Path
import sys
import io

# Đảm bảo in tiếng Việt không lỗi trên console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_json(path: str) -> dict:
    if not Path(path).exists():
        print(f"Lỗi: Không tìm thấy file {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    gestures = load_json("configs/gestures.json")
    if not gestures: return

    inv = {v.upper(): int(k) for k, v in gestures.items()}
    root_out_dir = Path("data/raw_videos")
    root_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Danh sách ký hiệu ---")
    for k, v in gestures.items():
        print(f" {k}: {v}")
    
    label_name = input("\nNhập tên ký hiệu (ví dụ: CHAO): ").strip().upper()
    if label_name not in inv:
        print("Ký hiệu không tồn tại trong configs/gestures.json")
        return

    # TẠO THƯ MỤC CON THEO NHÃN
    label_dir = root_out_dir / label_name
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không mở được camera")
        return

    fps = 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")

    print(f"\n--- ĐANG THU THẬP: {label_name} ---")
    print(f"Lưu tại: {label_dir}/")
    print("Bấm 'R' để Bắt đầu/Dừng ghi, 'Q' để Thoát")
    
    recording = False
    writer = None
    count = 0
    start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        disp = frame.copy()
        
        if recording:
            elapsed_time = time.time() - start_time
            color = (0, 0, 255)
            status = f"REC: {elapsed_time:.1f}s | Sample: {count+1}"
            if writer is not None: writer.write(frame)
        else:
            color = (0, 255, 0)
            status = f"IDLE | Ready for Sample {count+1}"

        cv2.putText(disp, f"LABEL: {label_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(disp, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Collect Data", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r') or key == ord('R'):
            if not recording:
                sample_id = int(time.time())
                video_path = label_dir / f"{label_name}_{sample_id}.avi"
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
                start_time = time.time()
                recording = True
                print(f"[*] Đang ghi mẫu {count+1}...")
            else:
                if writer is not None: writer.release()
                recording = False
                duration = time.time() - start_time
                count += 1
                print(f"[OK] Đã lưu mẫu {count} (Dài: {duration:.1f}s)")
        elif key == ord('q') or key == ord('Q'):
            if recording and writer is not None: writer.release()
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nHoàn tất! Đã thêm {count} video vào thư mục {label_name}.")

if __name__ == "__main__":
    main()
