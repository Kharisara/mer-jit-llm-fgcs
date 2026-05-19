import pandas as pd
import os

df = pd.read_csv("data/processed/meld_text_audio_video_arcface_states.csv")

for col in ["state_video_path", "video_embedding", "arcface_path"]:
    if col in df.columns:
        valid = df[col].apply(lambda x: isinstance(x, str) and os.path.exists(x)).mean()
        print(col, "exists ratio:", valid)
