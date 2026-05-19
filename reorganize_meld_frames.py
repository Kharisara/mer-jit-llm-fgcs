import os
import re
import shutil

FRAMES_ROOT = r"D:\mer-jit-llm\data\processed\MELD_frames"

# If your splits are different (e.g. "dev"), add them here
SPLITS = ["train", "val", "test"]

# dia89_utt10_2.jpg -> ("dia89_utt10", "2")
PATTERN = re.compile(r"^(dia\d+_utt\d+)_(\d+)\.jpg$", re.IGNORECASE)

def reorganize_split(split):
    split_dir = os.path.join(FRAMES_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f"[INFO] Split folder not found, skipping: {split_dir}")
        return

    print(f"[INFO] Processing split: {split_dir}")

    for fname in os.listdir(split_dir):
        fpath = os.path.join(split_dir, fname)
        if not os.path.isfile(fpath):
            continue

        m = PATTERN.match(fname)
        if not m:
            # skip anything that doesn't match diaX_uttY_N.jpg
            continue

        video_id = m.group(1)  # e.g. dia89_utt10
        target_dir = os.path.join(split_dir, video_id)
        os.makedirs(target_dir, exist_ok=True)

        new_path = os.path.join(target_dir, fname)
        if os.path.abspath(fpath) == os.path.abspath(new_path):
            continue

        shutil.move(fpath, new_path)

    print(f"[INFO] Done split: {split_dir}")

def main():
    for split in SPLITS:
        reorganize_split(split)

if __name__ == "__main__":
    main()
