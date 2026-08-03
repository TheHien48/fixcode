import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import json
from collections import deque


class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
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


def extract_frame_features(results, frame, mp_drawing, mp_hands) -> np.ndarray:
    """
    Tạo vector 132 chiều cho 1 frame live:
    - 0:66    = tay trái
    - 66:132  = tay phải
    """
    frame_features = np.zeros(132, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # "Left" hoặc "Right"

            mp_drawing.draw_landmarks(
                frame,
                lm,
                mp_hands.HAND_CONNECTIONS
            )

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


def main():
    with open("configs/gestures.json", "r", encoding="utf-8") as f:
        gestures_dict = json.load(f)

    actions = [gestures_dict[str(i)] for i in range(len(gestures_dict))]

    K = 16
    model_path = "models/gesture_classifier_v1.pt"

    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu")
    )

    model = MLP(
        in_dim=K * 132,
        n_classes=len(actions)
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Không mở được camera")

    frame_buffer = deque(maxlen=K)
    frame_count = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Không flip frame để đồng bộ với extract_2_hand.py.
            # Nếu muốn bật flip, phải đảo nhãn Left/Right.
            # frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            res_text = "..."
            conf_val = 0.0

            if results.multi_hand_landmarks:
                frame_features = extract_frame_features(
                    results,
                    frame,
                    mp_drawing,
                    mp_hands
                )

                if frame_count % 3 == 0:
                    frame_buffer.append(frame_features)

                if len(frame_buffer) == K:
                    input_data = np.array(
                        list(frame_buffer),
                        dtype=np.float32
                    ).reshape(1, -1)  # shape: (1, K * 132)

                    with torch.no_grad():
                        x = torch.from_numpy(input_data).float()
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)

                        conf, pred_id = torch.max(probs, dim=1)

                        pred_id = pred_id.item()
                        conf_val = conf.item()

                        if 0 <= pred_id < len(actions):
                            res_text = actions[pred_id]
                        else:
                            res_text = "UNKNOWN"

            else:
                frame_buffer.clear()

            h, w = frame.shape[:2]

            cv2.rectangle(
                frame,
                (0, 0),
                (w, 45),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                f"KQ: {res_text} ({conf_val * 100:.1f}%)",
                (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow("Test Gesture Live 132D", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                frame_buffer.clear()
                print("Đã xóa frame_buffer")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
