import os
import cv2
from tqdm import tqdm

# -------- CONFIG --------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "MELD")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "MELD_frames")

os.makedirs(OUT_DIR, exist_ok=True)

# How many frames per video to extract
NUM_FRAMES = 8


def extract_frames_for_split(split):
    split_video_dir = os.path.join(RAW_DIR, split, f"{split}_splits")
    split_out_dir = os.path.join(OUT_DIR, split)
    os.makedirs(split_out_dir, exist_ok=True)

    if not os.path.exists(split_video_dir):
        print(f"[ERROR] Missing directory: {split_video_dir}")
        return

    videos = sorted(os.listdir(split_video_dir))

    print(f"\n=== Processing {split.upper()} ({len(videos)} videos) ===")

    for vid_file in tqdm(videos):
        if not vid_file.endswith(".mp4"):
            continue

        in_path = os.path.join(split_video_dir, vid_file)
        out_prefix = os.path.join(split_out_dir, vid_file.replace(".mp4", ""))

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print(f"[WARNING] Cannot read: {in_path}, skipping.")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < NUM_FRAMES:
            frame_idxs = list(range(total_frames))
        else:
            frame_idxs = [int(i * total_frames / NUM_FRAMES) for i in range(NUM_FRAMES)]

        for i, frame_idx in enumerate(frame_idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = f"{out_prefix}_{i}.jpg"
            cv2.imwrite(out_path, frame)

        cap.release()


def main():
    for split in ["train", "dev", "test"]:
        extract_frames_for_split(split)
    print("\nDone extracting frames!")


if __name__ == "__main__":
    main()
