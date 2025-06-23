import cv2
import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# === CONFIGURATION ===
BROADCAST_FRAMES = "results/broadcast"
TACTICAM_FRAMES = "results/tacticam"
BROADCAST_PKL = "broadcast_features.pkl"
TACTICAM_PKL = "tacticam_features.pkl"
MAPPING = {
    4: 3,
    9: 2,
    15: 27
}
FRAME_INDEX = 100  # Choose a frame index where both players are likely visible

# === HELPER ===
def load_crop(frame_folder, player_id, frame_idx):
    frame_path = os.path.join(frame_folder, f"frame_{frame_idx:05d}.jpg")
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"❌ Frame not found: {frame_path}")
        return None

    model = YOLO("best.pt")
    results = model(frame)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in [0, 1]:  # Only players or goalkeepers
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

        # Optional: You could match by appearance or location here
        tracker = DeepSort()
        detections = [(bbox_xywh, float(box.conf[0]), 'player')]
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.track_id == player_id:
                crop = frame[y1:y2, x1:x2]
                return crop
    return None

# === VISUALIZATION ===
def show_side_by_side(crop1, crop2, label1="Tacticam", label2="Broadcast"):
    if crop1 is None or crop2 is None:
        print("❌ Could not find one or both crops.")
        return

    crop1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB)
    crop2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].imshow(crop1)
    ax[0].set_title(label1)
    ax[0].axis('off')

    ax[1].imshow(crop2)
    ax[1].set_title(label2)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

# === MAIN ===
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

print(f"🔍 Showing matches at frame index {FRAME_INDEX}")
for tacticam_id, broadcast_id in MAPPING.items():
    print(f"🧍 Tacticam ID {tacticam_id} ↔ Broadcast ID {broadcast_id}")
    crop1 = load_crop(TACTICAM_FRAMES, tacticam_id, FRAME_INDEX)
    crop2 = load_crop(BROADCAST_FRAMES, broadcast_id, FRAME_INDEX)
    show_side_by_side(crop1, crop2,
                      label1=f"Tacticam ID {tacticam_id}",
                      label2=f"Broadcast ID {broadcast_id}")
