# ⚽ Cross-Camera Player Identification using YOLOv8 + Deep SORT

This project identifies and matches football players across **two different camera angles** — `broadcast` and `tacticam` — by:

- Detecting players using **YOLOv8**
- Tracking them using **Deep SORT**
- Extracting feature embeddings using **ResNet**
- Matching players across views using **cosine similarity**
- Visually comparing matched player crops

---

## 🧠 Use Case

🔍 This system helps in analyzing the same players across multiple camera feeds — useful for **sports analytics**, **automated highlights**, or **tactical breakdowns**.

---

## 📁 Folder & File Structure

player-id-mapping/
├── detect_and_track.py # Detects & tracks players in both videos
├── extract_features.py # Extracts deep features from tracked player crops
├── match_players.py # Matches player IDs across the two views
├── visual_compare.py # Displays side-by-side visual confirmation
├── requirements.txt # Python dependencies
├── README.md # This file
├── best.pt # YOLOv8 model weights (NOT uploaded)
├── videos/
│ ├── broadcast.mp4 # Broadcast view input
│ └── tacticam.mp4 # Tacticam view input
└── results/
├── broadcast/
│ ├── frame_00001.jpg # Tracked + labeled broadcast frames
│ └── ...
├── tacticam/
│ ├── frame_00001.jpg # Tracked + labeled tacticam frames
│ └── ...
├── features/
│ ├── broadcast_features.npy
│ └── tacticam_features.npy
└── matched_players.json # Mapping of Tacticam ID → Broadcast ID


---

## 🚀 How to Run (Step-by-Step)

### 🔧 1. Install dependencies

```bash
pip install -r requirements.txt
