# tools/make_metadata_splits.py
import pandas as pd
import math
import os

src = "data/processed/meld_text_audio_video_arcface_states.csv"
out_dir = "data/processed/splits"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(src)
N = len(df)
chunk = 2000
num = math.ceil(N / chunk)
print(f"Total rows: {N}, chunk size: {chunk}, parts: {num}")
for i in range(num):
    start = i * chunk
    end = min(N, (i + 1) * chunk)
    sub = df.iloc[start:end].reset_index(drop=True)
    out_path = os.path.join(out_dir, f"meld_meta_chunk{i+1:02d}.csv")
    sub.to_csv(out_path, index=False)
    print(f"Saved {out_path} rows {start}:{end}")
