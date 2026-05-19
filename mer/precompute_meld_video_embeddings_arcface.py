# mer/precompute_meld_video_embeddings_arcface.py

import os
import argparse
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None
    print(
        "[WARN] insightface is not installed. "
        "Install it with: pip install insightface onnxruntime"
    )


def get_project_root() -> Path:
    this_dir = Path(__file__).resolve().parent  # .../mer
    return this_dir.parent                      # .../mer-jit-llm


def build_utterance_id(row) -> str:
    """
    MELD naming convention: dia{Dialogue_ID}_utt{Utterance_ID}
    These IDs should match your frame dir names.
    """
    d_id = int(row["Dialogue_ID"])
    u_id = int(row["Utterance_ID"])
    return f"dia{d_id}_utt{u_id}"


def get_split_folder(split_value: str) -> str:
    """
    Map CSV split -> frames subfolder name.
    Your CSV has 'train', 'dev', 'test'.
    Your frames root likely has 'train', 'dev', 'test'.
    If dev frames were merged into 'train', you can map 'dev' -> 'train' here.
    """
    split_value = str(split_value).lower()
    if split_value in ["train", "dev", "test"]:
        return split_value
    # fallback
    return "train"


def init_arcface(device: str = "cpu"):
    if FaceAnalysis is None:
        raise ImportError(
            "insightface not available. Run: pip install insightface onnxruntime"
        )

    # ctx_id = -1 -> CPU; >=0 -> GPU index
    if device == "cpu":
        ctx_id = -1
    else:
        ctx_id = 0

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def extract_arcface_embedding_from_frames(
    face_app,
    frame_paths: List[Path],
    emb_dim: int = 512,
) -> Optional[np.ndarray]:
    """
    Given a list of frame image paths, run ArcFace on faces and
    average their normed embeddings.

    Returns:
        np.ndarray of shape [emb_dim], or None if no faces found.
    """
    all_embs = []

    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # BGR -> RGB is optional; InsightFace generally expects BGR from cv2,
        # but works fine as is for most configs.
        faces = face_app.get(img)
        if len(faces) == 0:
            continue

        # pick the largest face (by bbox area)
        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        # normed_embedding is already L2-normalized & 512-d
        emb = best_face.normed_embedding
        if emb is None:
            continue

        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape[0] != emb_dim:
            # unexpected; skip
            continue

        all_embs.append(emb)

    if len(all_embs) == 0:
        return None

    all_embs = np.stack(all_embs, axis=0)  # [N, 512]
    mean_emb = all_embs.mean(axis=0)
    # re-normalize to unit length
    norm = np.linalg.norm(mean_emb) + 1e-12
    mean_emb = mean_emb / norm
    return mean_emb.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=512,
        help="ArcFace embedding dimension (buffalo_l is 512).",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    data_processed = project_root / "data" / "processed"

    input_csv = data_processed / "meld_text_audio.csv"
    # we will create a new CSV that includes video_embedding_path from ArcFace
    output_csv = data_processed / "meld_text_audio_video_arcface.csv"

    frames_root = data_processed / "MELD_frames"
    embeddings_root = data_processed / "MELD_video_embeddings_arcface"
    embeddings_root.mkdir(parents=True, exist_ok=True)

    print("Using device:", args.device)
    print("Input CSV:", input_csv)
    print("Output CSV:", output_csv)
    print("Frames root:", frames_root)
    print("Embeddings root:", embeddings_root)

    df = pd.read_csv(input_csv)
    print("CSV columns:", list(df.columns))

    # Initialize ArcFace
    face_app = init_arcface(device=args.device)

    video_paths = []
    num_rows = len(df)

    for idx, row in tqdm(df.iterrows(), total=num_rows, desc="Encoding video (ArcFace)"):
        split = row["split"]
        split_folder = get_split_folder(split)

        utt_id = build_utterance_id(row)  # e.g., dia339_utt1
        frame_dir = frames_root / split_folder / utt_id

        if not frame_dir.exists():
            # no frames for this utterance
            video_paths.append(np.nan)
            continue

        # collect frames (jpg/png)
        frame_files = sorted(
            list(frame_dir.glob("*.jpg")) + list(frame_dir.glob("*.png"))
        )
        if len(frame_files) == 0:
            video_paths.append(np.nan)
            continue

        emb = extract_arcface_embedding_from_frames(
            face_app,
            frame_files,
            emb_dim=args.emb_dim,
        )

        if emb is None:
            # no faces found => mark missing
            video_paths.append(np.nan)
            continue

        # save as .npy
        split_dir = embeddings_root / split_folder
        split_dir.mkdir(parents=True, exist_ok=True)

        emb_path = split_dir / f"{utt_id}.npy"
        np.save(emb_path, emb)

        # store relative path to keep CSV portable
        rel_path = os.path.relpath(emb_path, project_root)
        video_paths.append(rel_path)

    df["video_embedding_path"] = video_paths
    df.to_csv(output_csv, index=False)
    print("Done. Saved updated CSV with ArcFace paths to:", output_csv)

    valid = df["video_embedding_path"].notna().sum()
    print(f"Rows with ArcFace embeddings: {valid}/{len(df)}")


if __name__ == "__main__":
    main()
