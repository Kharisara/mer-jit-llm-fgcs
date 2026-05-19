"""
extract_tav_context_states.py

Goal:
 - Guarantee a 512-D state vector for every MELD utterance.
 - Priority:
     1) video embedding
     2) text + audio
     3) text only
     4) audio only
 - Log modality availability and state source.
"""

import os
import argparse
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional deps
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except Exception:
    LIBROSA_AVAILABLE = False


STATE_DIM = 512
TEXT_DIM = 256
AUDIO_DIM = 256
DEFAULT_TEXT_MODEL = "distilbert-base-uncased"


# ---------------- Utilities ----------------
def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def is_valid_path(x) -> bool:
    return isinstance(x, str) and len(x) > 0 and not pd.isna(x)


def load_vector_if_exists(path: str) -> Optional[np.ndarray]:
    if not is_valid_path(path) or not os.path.exists(path):
        return None
    try:
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr).ravel().astype(np.float32)
    except Exception:
        return None


# ---------------- Text encoder ----------------
class TextEncoder:
    def __init__(self, model_name=DEFAULT_TEXT_MODEL):
        self.available = False
        if not TRANSFORMERS_AVAILABLE:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False

    def encode(self, text: str) -> np.ndarray:
        if not self.available or not isinstance(text, str) or len(text.strip()) == 0:
            return np.zeros(768, dtype=np.float32)
        import torch
        with torch.no_grad():
            enc = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=128,
                return_tensors="pt",
            )
            out = self.model(**enc)
            return out.last_hidden_state[:, 0].numpy().squeeze(0).astype(np.float32)


# ---------------- Audio encoder ----------------
def audio_logmel(audio_path: Optional[str], sr=16000, n_mels=64, max_dur=3.0):
    if not LIBROSA_AVAILABLE or not is_valid_path(audio_path) or not os.path.exists(audio_path):
        return np.zeros(n_mels, dtype=np.float32)
    try:
        wav, _ = librosa.load(audio_path, sr=sr, mono=True)
        max_len = int(sr * max_dur)
        wav = wav[:max_len] if len(wav) > max_len else np.pad(wav, (0, max_len - len(wav)))
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
        logmel = librosa.power_to_db(mel + 1e-6)
        return np.mean(logmel, axis=1).astype(np.float32)
    except Exception:
        return np.zeros(n_mels, dtype=np.float32)


def expand_audio(mel: np.ndarray) -> np.ndarray:
    if np.allclose(mel, 0.0):
        return np.zeros(AUDIO_DIM, dtype=np.float32)
    return np.interp(
        np.linspace(0, len(mel) - 1, AUDIO_DIM),
        np.arange(len(mel)),
        mel
    ).astype(np.float32)


def build_state(text_vec: np.ndarray, audio_vec: np.ndarray) -> np.ndarray:
    text_part = text_vec[:TEXT_DIM] if len(text_vec) >= TEXT_DIM else np.pad(text_vec, (0, TEXT_DIM - len(text_vec)))
    audio_part = expand_audio(audio_vec)
    return np.concatenate([text_part, audio_part]).astype(np.float32)


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    safe_mkdir(args.state_root)
    df = pd.read_csv(args.csv)

    # ensure columns
    df["state_path"] = df.get("state_path", "")
    df["state_source"] = ""
    df["has_text"] = False
    df["has_audio"] = False
    df["has_video"] = False

    text_encoder = TextEncoder()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting states"):
        if is_valid_path(row["state_path"]) and os.path.exists(row["state_path"]) and not args.overwrite:
            continue

        # ---- VIDEO ----
        video_path = row.get("state_video_path") or row.get("video_embedding") or row.get("arcface_path")
        vec = load_vector_if_exists(video_path) if is_valid_path(video_path) else None

        if vec is not None:
            state = vec[:STATE_DIM] if len(vec) >= STATE_DIM else np.pad(vec, (0, STATE_DIM - len(vec)))
            source = "video"
            has_text = has_audio = False
            has_video = True
        else:
            # ---- FALLBACK ----
            text = row.get("text")
            audio_path = row.get("audio_path") or row.get("wav_path")

            text_vec = text_encoder.encode(text)
            audio_vec = audio_logmel(audio_path)

            has_text = not np.allclose(text_vec, 0.0)
            has_audio = not np.allclose(audio_vec, 0.0)
            has_video = False

            if not (has_text or has_audio):
                continue

            state = build_state(text_vec, audio_vec)
            source = "text_audio" if has_text and has_audio else "text_only" if has_text else "audio_only"

        # ---- SAVE ----
        fname = f"d{int(row['Dialogue_ID'])}_u{int(row['Utterance_ID'])}.npy"
        fpath = os.path.join(args.state_root, fname)
        np.save(fpath, state)

        df.at[idx, "state_path"] = fpath
        df.at[idx, "state_source"] = source
        df.at[idx, "has_text"] = has_text
        df.at[idx, "has_audio"] = has_audio
        df.at[idx, "has_video"] = has_video

    # save
    backup = args.csv + ".bak"
    if not os.path.exists(backup):
        df.to_csv(backup, index=False)
    df.to_csv(args.csv, index=False)

    print("\n✔ Extraction complete")
    print("Total states:", df["state_path"].notna().sum())


if __name__ == "__main__":
    main()
