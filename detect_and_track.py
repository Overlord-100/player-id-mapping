from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import os
from utils.visualization import draw_tracks, save_frame

def process_video(video_path, model_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)  # ✅ Automatically creates output folder

    model = YOLO(model_path)
    tracker = DeepSort(max_age=30)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]  # YOLO inference
        detections = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Assuming class 0 = player
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                bbox_xywh = [x1, y1, x2 - x1, y2 - y1]  # Convert to (x, y, w, h)
                detections.append((bbox_xywh, conf, 'player'))

        tracks = tracker.update_tracks(detections, frame=frame)
        frame = draw_tracks(frame, tracks)
        save_frame(frame, os.path.join(save_dir, f"frame_{frame_idx:05d}.jpg"))
        frame_idx += 1

    cap.release()
    print(f"✅ Done processing {video_path}. {frame_idx} frames saved to {save_dir}")

# 🔽 MAIN EXECUTION (You can call this for either video)
if __name__ == "__main__":
    process_video(
        video_path="broadcast.mp4",
        model_path="best.pt",
        save_dir="results/broadcast"
    )

    process_video(
        video_path="tacticam.mp4",
        model_path="best.pt",
        save_dir="results/tacticam"
    )
