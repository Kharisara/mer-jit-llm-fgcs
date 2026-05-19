import pandas as pd

df = pd.read_csv("data/processed/meld_text_audio_video_arcface_states.csv")

print("\nState source breakdown:")
print(df["state_source"].value_counts())

print("\nModality availability (mean):")
print(df[["has_text", "has_audio", "has_video"]].mean())
