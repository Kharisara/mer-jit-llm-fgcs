import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from .video_encoder import VideoEncoder


def build_paths():
    # project_root = .../mer-jit-llm
    project_root = Path(__file__).resolve().parents[1]

    data_processed = project_root / "data" / "processed"

    input_csv = data_processed / "meld_text_audio.csv"
    output_csv = data_processed / "meld_text_audio_video.csv"

    frames_root = data_processed / "MELD_frames"
    embeddings_root = data_processed / "MELD_video_embeddings"

    return project_root, input_csv, output_csv, frames_root, embeddings_root


def main():
    print("DEBUG: running SIMPLE precompute_meld_video_embeddings (no infer_video_id)")  # << marker

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    project_root, input_csv, output_csv, frames_root, embeddings_root = build_paths()

    print(f"Input CSV: {input_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Frames root: {frames_root}")
    print(f"Embeddings root: {embeddings_root}")

    df = pd.read_csv(input_csv)
    print(f"CSV columns: {list(df.columns)}")

    # Make sure required columns exist
    required_cols = ["split", "Dialogue_ID", "Utterance_ID"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Required column '{c}' not found in CSV.")

    # Initialize video encoder
    encoder = VideoEncoder(device=device, max_frames=16)

    # New column for embedding path
    if "video_embedding_path" not in df.columns:
        df["video_embedding_path"] = ""

    total_rows = len(df)
    used_rows = 0
    skipped_rows = 0

    embeddings_root.mkdir(parents=True, exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Encoding video"):
        split = str(row["split"]).strip()

        # Build video id like dia0_utt0, dia1_utt3 etc.
        try:
            dia_id = int(row["Dialogue_ID"])
            utt_id = int(row["Utterance_ID"])
        except Exception:
            skipped_rows += 1
            continue

        video_id = f"dia{dia_id}_utt{utt_id}"

        # Frames dir: <frames_root>/<split>/diaX_uttY
        frame_dir = frames_root / split / video_id
        if not frame_dir.is_dir():
            skipped_rows += 1
            continue

        # Embedding file path
        split_embed_dir = embeddings_root / split
        split_embed_dir.mkdir(parents=True, exist_ok=True)
        emb_path = split_embed_dir / f"{video_id}.npy"

        # If already exists, you can skip recomputing (optional)
        if emb_path.is_file():
            df.at[idx, "video_embedding_path"] = str(emb_path.relative_to(project_root))
            used_rows += 1
            continue

        # Encode frames
        embedding = encoder.encode_frames_in_dir(str(frame_dir))
        if embedding is None:
            skipped_rows += 1
            continue

        # Save embedding
        np.save(emb_path, embedding)

        # Store relative path in CSV
        df.at[idx, "video_embedding_path"] = str(emb_path.relative_to(project_root))
        used_rows += 1

    print()
    print(f"Total rows in original CSV: {total_rows}")
    print(f"Rows with video embeddings: {used_rows}")
    print(f"Skipped rows (no frames / errors): {skipped_rows}")

    if used_rows == 0:
        print("[ERROR] No video embeddings were created. Check your frames directory layout.")
    else:
        print("[OK] Some video embeddings were created.")

    # Save updated CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved updated CSV with video_embedding_path to: {output_csv}")


if __name__ == "__main__":
    main()
