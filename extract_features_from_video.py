import os
import cv2
import pickle
from PIL import Image
from utils.extract_features import extract_features
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ⚠️ Set to either 'broadcast' or 'tacticam'
INPUT_VIDEO = "broadcast"

FRAME_FOLDER = f"results/{INPUT_VIDEO}/"
OUTPUT_FEATURE_FILE = f"{INPUT_VIDEO}_features.pkl"

model = YOLO("best.pt")
tracker = DeepSort(max_age=30)
player_features = {}

frame_files = sorted(os.listdir(FRAME_FOLDER))
print(f"🔍 Processing {len(frame_files)} frames in: {FRAME_FOLDER}")

for idx, fname in enumerate(frame_files):
    frame_path = os.path.join(FRAME_FOLDER, fname)
    frame = cv2.imread(frame_path)

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        # ✅ Accept both players and goalkeepers
        if cls in [0, 1]:  # 0 = player, 1 = goalkeeper (adjust as per your model)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox_xywh, conf, 'player'))

    print(f"[Frame {idx}] → {len(detections)} players detected")

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        # 🛡️ Validate crop bounds
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            continue
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            continue

        try:
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            feat = extract_features(pil_crop)

            if track_id not in player_features:
                player_features[track_id] = []
            player_features[track_id].append(feat)
        except Exception as e:
            print(f"⚠️ Skipping track {track_id}: {e}")

# Final aggregation
print("📊 Averaging features for each player...")
final_features = {}
for pid, feats in player_features.items():
    if len(feats) > 0:
        final_features[pid] = sum(feats) / len(feats)

print(f"✅ Extracted features for {len(final_features)} players.")

# Save to file
with open(OUTPUT_FEATURE_FILE, "wb") as f:
    pickle.dump(final_features, f)

print(f"💾 Saved to: {OUTPUT_FEATURE_FILE}")
