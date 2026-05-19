import os
import subprocess
from tqdm import tqdm

# UPDATE THIS to your actual ffmpeg.exe path
FFMPEG_EXE = r"C:\Users\Kesara\Downloads\ffmpeg-2025-11-27-git-61b034a47c-full_build\bin\ffmpeg.exe"

RAW_DIR = r"D:\mer-jit-llm\data\raw\MELD"
OUT_DIR = r"D:\mer-jit-llm\data\processed\MELD_audio"

splits = ["train", "dev", "test"]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_audio():
    for split in splits:
        print(f"\n=== Processing {split.upper()} ===")

        video_dir = os.path.join(RAW_DIR, split, f"{split}_splits")
        out_dir = os.path.join(OUT_DIR, f"{split}_wav")
        ensure_dir(out_dir)

        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"Video directory not found: {video_dir}")

        files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
        files.sort()

        for fname in tqdm(files, desc=f"{split}"):
            in_path = os.path.join(video_dir, fname)
            out_name = fname.replace(".mp4", ".wav")
            out_path = os.path.join(out_dir, out_name)

            # Skip if already done
            if os.path.exists(out_path):
                continue

            cmd = [
                FFMPEG_EXE,
                "-i", in_path,
                "-ac", "1",
                "-ar", "16000",
                "-loglevel", "error",
                out_path,
            ]

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError:
                # Just warn and continue with next file
                print(f"\n[WARNING] Failed to process {in_path}, skipping.\n")
                if os.path.exists(out_path):
                    # remove partial/broken file if created
                    os.remove(out_path)
                continue

    print("\nDone! All possible audio extracted.")


if __name__ == "__main__":
    extract_audio()
