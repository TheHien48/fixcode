import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

mp_hands = mp.solutions.hands


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa 1 bàn tay thành vector 66 chiều:
    - 63 chiều: tọa độ landmarks tương đối đã scale
    - 3 chiều: vị trí cổ tay tuyệt đối
    """
    wrist_global = landmarks[0].copy()

    wrist = landmarks[0].copy()
    centered = landmarks - wrist

    ref = np.linalg.norm(centered[9])
    if ref < 1e-6:
        ref = 1.0

    scaled = centered / ref
    shape_features = scaled.reshape(-1)

    combined = np.concatenate([shape_features, wrist_global])
    return combined.astype(np.float32)


def extract_frame_features(results) -> np.ndarray:
    """
    Tạo vector 132 chiều cho 1 frame:
    - 0:66    = tay trái
    - 66:132  = tay phải
    """
    frame_features = np.zeros(132, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" hoặc "Right"

            pts = np.array(
                [[p.x, p.y, p.z] for p in lm.landmark],
                dtype=np.float32
            )

            norm_pts = normalize_landmarks(pts)

            if label == "Left":
                frame_features[:66] = norm_pts
            elif label == "Right":
                frame_features[66:] = norm_pts

    return frame_features


def process_video(video_path: Path, out_dir: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Không mở được video: {video_path}")
        return

    seq = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Không flip frame để nhãn Left/Right đồng bộ khi train và test
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                frame_features = extract_frame_features(results)
                seq.append(frame_features)

    cap.release()

    if len(seq) == 0:
        print(f"Không phát hiện tay trong video: {video_path}")
        return

    arr = np.stack(seq, axis=0)  # shape: (T, 132)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (video_path.stem + ".npy")
    np.save(out_path, arr)

    print(f"Đã lưu: {out_path} | shape = {arr.shape}")


def main():
    raw_dir = Path("data/raw_videos")
    out_dir = Path("data/keypoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    video_exts = [".avi", ".mp4", ".mov", ".mkv"]

    vids = sorted([
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in video_exts
    ])

    print(f"Tìm thấy {len(vids)} video.")
    print("Đang trích xuất keypoints 132 chiều cho 1 tay / 2 tay...")

    for vp in vids:
        process_video(vp, out_dir)

    print("\n--- Xong trích xuất keypoints 132 chiều ---")


if __name__ == "__main__":
    main()
