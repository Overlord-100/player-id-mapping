import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === CONFIG ===
BROADCAST_FRAMES_DIR = "results/broadcast"
TACTICAM_FRAMES_DIR = "results/tacticam"
MODEL_PATH = "best.pt"
FRAME_INDEX = 100  # Choose a good frame (try different values)

# === Paste the mapping you got from match_players.py ===
# Format: {tacticam_id: broadcast_id}
MAPPING = {
    4: 3,
    9: 2,
    15: 27
}

model = YOLO(MODEL_PATH)

def get_highest_conf_crop(frame_path, model):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"❌ Could not load: {frame_path}")
        return None

    results = model(frame)[0]
    best_crop = None
    max_conf = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in [0, 1]:  # Only players + goalkeepers
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        conf = float(box.conf[0])
        if conf > max_conf:
            max_conf = conf
            best_crop = crop

    return best_crop

def visualize_pair(tacti_crop, broad_crop, tacti_id, broad_id):
    if tacti_crop is None or broad_crop is None:
        print(f"⚠️ Skipping: one or both crops not found.")
        return

    tacti_crop = cv2.cvtColor(tacti_crop, cv2.COLOR_BGR2RGB)
    broad_crop = cv2.cvtColor(broad_crop, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(tacti_crop)
    axes[0].set_title(f"Tacticam ID {tacti_id}")
    axes[0].axis('off')

    axes[1].imshow(broad_crop)
    axes[1].set_title(f"Broadcast ID {broad_id}")
    axes[1].axis('off')

    plt.suptitle("Matched Player Visual Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()

# === MAIN LOOP ===
print(f"🔍 Visualizing matched players from frame {FRAME_INDEX}")
for tacticam_id, broadcast_id in MAPPING.items():
    tacti_frame = os.path.join(TACTICAM_FRAMES_DIR, f"frame_{FRAME_INDEX:05d}.jpg")
    broad_frame = os.path.join(BROADCAST_FRAMES_DIR, f"frame_{FRAME_INDEX:05d}.jpg")

    tacti_crop = get_highest_conf_crop(tacti_frame, model)
    broad_crop = get_highest_conf_crop(broad_frame, model)

    visualize_pair(tacti_crop, broad_crop, tacticam_id, broadcast_id)
