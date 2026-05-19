# # scripts/preprocess_meld.py

# import os
# import pandas as pd

# # Paths (adjust if your structure is different)
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "MELD")
# OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# os.makedirs(OUT_DIR, exist_ok=True)

# # Input CSVs (MELD standard names)
# TRAIN_CSV = os.path.join(RAW_DIR, "train_sent_emo.csv")
# DEV_CSV = os.path.join(RAW_DIR, "dev_sent_emo.csv")
# TEST_CSV = os.path.join(RAW_DIR, "test_sent_emo.csv")

# # Output
# OUT_COMBINED = os.path.join(OUT_DIR, "meld_text_only.csv")


# # ---- Helper: load and standardize one split ----
# def load_split(path, split_name):
#     """
#     Expected MELD columns typically include:
#       - Dialogue_ID
#       - Utterance_ID
#       - Utterance
#       - Emotion
#       - Sentiment
#       - Speaker
#       ...
#     We only care about: Dialogue_ID, Utterance_ID, Utterance, Emotion
#     """
#     print(f"Loading {split_name} from {path}")
#     df = pd.read_csv(path)

#     required_cols = ["Dialogue_ID", "Utterance_ID", "Utterance", "Emotion"]
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Column '{col}' not found in {path}. Available: {df.columns.tolist()}")

#     df = df[required_cols].copy()
#     df["split"] = split_name
#     return df


# def main():
#     # Load splits
#     train_df = load_split(TRAIN_CSV, "train")
#     dev_df = load_split(DEV_CSV, "dev")
#     test_df = load_split(TEST_CSV, "test")

#     # Combine
#     full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

#     # Map MELD emotions to your 4-class scheme
#     emotion_map = {
#         "joy": "happy",
#         "anger": "angry",
#         "neutral": "neutral",
#         "sadness": "sad",
#         # others we will drop
#     }

#     full_df["MappedEmotion"] = full_df["Emotion"].map(emotion_map)

#     # Drop rows with emotions we don't use (fear, disgust, surprise, etc.)
#     before = len(full_df)
#     full_df = full_df[full_df["MappedEmotion"].notna()].copy()
#     after = len(full_df)
#     print(f"Dropped {before - after} samples with unsupported emotions.")

#     # Build a unique utterance_id string: e.g. "dialogue_3_utt_5"
#     full_df["utterance_id"] = (
#         "d" + full_df["Dialogue_ID"].astype(str) + "_u" + full_df["Utterance_ID"].astype(str)
#     )

#     # Rename columns to match your training script expectations
#     out_df = full_df.rename(
#         columns={
#             "Utterance": "text",
#             "MappedEmotion": "label",
#         }
#     )[["utterance_id", "text", "label", "split"]]

#     print("Label distribution:")
#     print(out_df["label"].value_counts())

#     # Save combined CSV
#     out_df.to_csv(OUT_COMBINED, index=False, encoding="utf-8")
#     print(f"\nSaved combined MELD text-only CSV to: {OUT_COMBINED}")


# if __name__ == "__main__":
#     main()

# scripts/preprocess_meld.py

# import os
# import pandas as pd

# # Paths
# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "MELD")
# OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
# os.makedirs(OUT_DIR, exist_ok=True)

# # Input CSVs
# TRAIN_CSV = os.path.join(RAW_DIR, "train_sent_emo.csv")
# DEV_CSV = os.path.join(RAW_DIR, "dev_sent_emo.csv")
# TEST_CSV = os.path.join(RAW_DIR, "test_sent_emo.csv")

# # Output
# OUT_TEXT_ONLY = os.path.join(OUT_DIR, "meld_text_only.csv")
# OUT_TEXT_AUDIO = os.path.join(OUT_DIR, "meld_text_audio.csv")


# def load_split(path, split_name):
#     print(f"Loading {split_name} from {path}")
#     df = pd.read_csv(path)

#     required_cols = ["Dialogue_ID", "Utterance_ID", "Utterance", "Emotion"]
#     for col in required_cols:
#         if col not in df.columns:
#             raise ValueError(f"Column '{col}' not found in {path}. Available: {df.columns.tolist()}")

#     df = df[required_cols].copy()
#     df["split"] = split_name
#     return df


# def add_audio_path(df, raw_dir):
#     """
#     Add an audio_path column based on Dialogue_ID and Utterance_ID.
#     Assumes files are named like: dia<Dialogue_ID>_utt<Utterance_ID>.wav
#     and stored in:
#       raw_dir/train_wav/
#       raw_dir/dev_wav/
#       raw_dir/test_wav/
#     """
#     def path_for_row(row):
#         split = row["split"]
#         dia = row["Dialogue_ID"]
#         utt = row["Utterance_ID"]
#         filename = f"dia{dia}_utt{utt}.wav"
#         # adjust folders here if your structure is different
#         if split == "train":
#             folder = "train_wav"
#         elif split == "dev":
#             folder = "dev_wav"
#         else:
#             folder = "test_wav"
#         return os.path.join(raw_dir, folder, filename)

#     df["audio_path"] = df.apply(path_for_row, axis=1)
#     return df


# def main():
#     # Load splits
#     train_df = load_split(TRAIN_CSV, "train")
#     dev_df = load_split(DEV_CSV, "dev")
#     test_df = load_split(TEST_CSV, "test")

#     # Combine
#     full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

#     # Map MELD emotions to 4-class scheme
#     emotion_map = {
#         "joy": "happy",
#         "anger": "angry",
#         "neutral": "neutral",
#         "sadness": "sad",
#         # others dropped
#     }

#     full_df["MappedEmotion"] = full_df["Emotion"].map(emotion_map)

#     # Drop unsupported emotions
#     before = len(full_df)
#     full_df = full_df[full_df["MappedEmotion"].notna()].copy()
#     after = len(full_df)
#     print(f"Dropped {before - after} samples with unsupported emotions.")

#     # Unique utterance_id
#     full_df["utterance_id"] = (
#         "d" + full_df["Dialogue_ID"].astype(str) + "_u" + full_df["Utterance_ID"].astype(str)
#     )

#     # Text-only dataframe
#     text_df = full_df.rename(
#         columns={
#             "Utterance": "text",
#             "MappedEmotion": "label",
#         }
#     )[["utterance_id", "text", "label", "split"]]

#     print("Label distribution (text-only):")
#     print(text_df["label"].value_counts())
#     text_df.to_csv(OUT_TEXT_ONLY, index=False, encoding="utf-8")
#     print(f"\nSaved MELD text-only CSV to: {OUT_TEXT_ONLY}")

#     # Text + audio dataframe with audio_path
#     ta_df = full_df.rename(
#         columns={
#             "Utterance": "text",
#             "MappedEmotion": "label",
#         }
#     )[["utterance_id", "text", "label", "split", "Dialogue_ID", "Utterance_ID"]]

#     ta_df = add_audio_path(ta_df, RAW_DIR)
#     print("\nExample audio paths:")
#     print(ta_df[["utterance_id", "audio_path"]].head())

#     ta_df.to_csv(OUT_TEXT_AUDIO, index=False, encoding="utf-8")
#     print(f"\nSaved MELD text+audio CSV to: {OUT_TEXT_AUDIO}")


# if __name__ == "__main__":
#     main()

import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "MELD")
AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "MELD_audio")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Root-level CSVs (from MELD-RAW)
TRAIN_CSV = os.path.join(RAW_DIR, "train_sent_emo.csv")
DEV_CSV = os.path.join(RAW_DIR, "dev_sent_emo.csv")
TEST_CSV = os.path.join(RAW_DIR, "test_sent_emo.csv")

OUT_TEXT_ONLY = os.path.join(OUT_DIR, "meld_text_only.csv")
OUT_TEXT_AUDIO = os.path.join(OUT_DIR, "meld_text_audio.csv")


def load_split(path, split_name):
    print(f"Loading {split_name} from {path}")
    df = pd.read_csv(path)

    required_cols = ["Dialogue_ID", "Utterance_ID", "Utterance", "Emotion"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {path}. Available: {df.columns.tolist()}")

    df = df[required_cols].copy()
    df["split"] = split_name
    return df


def build_audio_path(row):
    """
    Map (split, Dialogue_ID, Utterance_ID) -> WAV file path.
    We assume filenames like dia<Dialogue_ID>_utt<Utterance_ID>.wav
    inside:
      data/processed/MELD_audio/train_wav/
      data/processed/MELD_audio/dev_wav/
      data/processed/MELD_audio/test_wav/
    """
    split = row["split"]
    dia = row["Dialogue_ID"]
    utt = row["Utterance_ID"]

    if split == "train":
        sub = "train_wav"
    elif split == "dev":
        sub = "dev_wav"
    else:
        sub = "test_wav"

    filename = f"dia{dia}_utt{utt}.wav"
    return os.path.join(AUDIO_DIR, sub, filename)


def main():
    # 1. Load all splits
    train_df = load_split(TRAIN_CSV, "train")
    dev_df = load_split(DEV_CSV, "dev")
    test_df = load_split(TEST_CSV, "test")

    full_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

    # 2. Map emotions -> 4 classes
    emotion_map = {
        "joy": "happy",
        "anger": "angry",
        "neutral": "neutral",
        "sadness": "sad",
    }

    full_df["MappedEmotion"] = full_df["Emotion"].map(emotion_map)
    before = len(full_df)
    full_df = full_df[full_df["MappedEmotion"].notna()].copy()
    after = len(full_df)
    print(f"Dropped {before - after} samples with unsupported emotions.")

    # 3. Build utterance_id
    full_df["utterance_id"] = (
        "d" + full_df["Dialogue_ID"].astype(str) + "_u" + full_df["Utterance_ID"].astype(str)
    )

    # ---------- TEXT-ONLY ----------
    text_df = full_df.rename(
        columns={
            "Utterance": "text",
            "MappedEmotion": "label",
        }
    )[["utterance_id", "text", "label", "split"]]

    print("\nLabel distribution (TEXT-ONLY):")
    print(text_df["label"].value_counts())
    text_df.to_csv(OUT_TEXT_ONLY, index=False, encoding="utf-8")
    print(f"\nSaved MELD text-only CSV to: {OUT_TEXT_ONLY}")

    # ---------- TEXT + AUDIO ----------
    ta_df = full_df.rename(
        columns={
            "Utterance": "text",
            "MappedEmotion": "label",
        }
    )[["utterance_id", "text", "label", "split", "Dialogue_ID", "Utterance_ID"]]

    ta_df["audio_path"] = ta_df.apply(build_audio_path, axis=1)

    # Keep only rows where the wav actually exists
    exists_mask = ta_df["audio_path"].apply(os.path.exists)
    missing = (~exists_mask).sum()
    if missing > 0:
        print(f"\nWarning: {missing} rows have missing audio files and will be dropped.")
    ta_df = ta_df[exists_mask].reset_index(drop=True)

    print("\nLabel distribution (TEXT + AUDIO, after dropping missing audio):")
    print(ta_df["label"].value_counts())

    print("\nExample rows:")
    print(ta_df[["utterance_id", "split", "audio_path"]].head())

    ta_df.to_csv(OUT_TEXT_AUDIO, index=False, encoding="utf-8")
    print(f"\nSaved MELD text+audio CSV to: {OUT_TEXT_AUDIO}")


if __name__ == "__main__":
    main()
