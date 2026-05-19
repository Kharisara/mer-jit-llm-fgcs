"""
mer/simulate_dialogue.py

End-to-end simulation of MER + JITAI + LLM (Ollama).

Pipeline:
- Load MELD metadata CSV with state_path
- Load pre-fit PCA (e.g., models/pca_states_2.pkl)
- Build a policy-constrained prompt
- Call local Ollama (gemma3:1b)
- Save outputs to CSV for analysis

Output columns:
dialogue_id, utt_id, text, label, valence, action, z_x, z_y, llm_reply

Run (from project root):
python -m mer.simulate_dialogue
"""

import os
import csv
import argparse
import requests
import numpy as np
import pandas as pd
import joblib

# -------------------------------
# Ollama config
# -------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"


# -------------------------------
# Utilities
# -------------------------------
def load_pca(pca_path: str):
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA file not found: {pca_path}")
    return joblib.load(pca_path)


def build_prompt(
    text: str,
    label: str,
    valence: float,
    zproj: np.ndarray,
    action: int,
    history: list[str],
) -> str:
    """
    Build a STRICT, policy-controlled prompt.
    """

    if action == 1:
        instruction = (
            "Provide emotional validation and support. "
            "Acknowledge the feeling and offer ONE small, practical coping suggestion."
        )
    else:
        instruction = (
            "Respond neutrally and minimally. "
            "Do NOT provide emotional validation, reassurance, or coping advice."
        )

    hist_block = ""
    if history:
        hist_block = "\nRecent dialogue (context only):\n" + "\n".join(history[-2:])

    return f"""
You are an assistant operating under a decision policy.

User utterance:
"{text}"

Detected emotion label: {label}
Valence score: {valence}

Affective state summary (PCA projection):
x = {zproj[0]:.3f}, y = {zproj[1]:.3f}

Policy action:
{action}  (0 = no intervention, 1 = supportive intervention)

INSTRUCTION (MANDATORY):
{instruction}

Rules:
- Be concise (2–4 sentences max).
- No diagnoses.
- No medical advice.
- No self-harm instructions.
- Follow the policy action strictly.

{hist_block}
""".strip()


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 120,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()


# -------------------------------
# Main simulation
# -------------------------------
def simulate(metadata_csv, pca_path, out_csv, max_rows=None):
    df = pd.read_csv(metadata_csv)
    pca = load_pca(pca_path)

    print(f"[simulate] Rows in CSV: {len(df)}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dialogue_id",
                "utt_id",
                "text",
                "label",
                "valence",
                "action",
                "z_x",
                "z_y",
                "llm_reply",
            ]
        )

    dialogue_history: dict[int, list[str]] = {}
    processed = 0

    for _, row in df.iterrows():
        if max_rows is not None and processed >= max_rows:
            break

        # Required fields
        dlg = int(row["Dialogue_ID"])
        utt = int(row["Utterance_ID"])
        text = str(row["text"])
        label = str(row["label"])
        valence = float(row.get("valence", 0.0))
        action = int(row.get("action", 0))
        state_path = row.get("state_path", None)

        if dlg not in dialogue_history:
            dialogue_history[dlg] = []

        # Load MER state
        if isinstance(state_path, str) and os.path.exists(state_path):
            try:
                z = np.load(state_path).ravel()
            except Exception:
                z = np.zeros(pca.components_.shape[1])
        else:
            z = np.zeros(pca.components_.shape[1])

        # Ensure correct dimension
        expected_dim = pca.components_.shape[1]
        if z.size < expected_dim:
            z = np.pad(z, (0, expected_dim - z.size))
        elif z.size > expected_dim:
            z = z[:expected_dim]

        zproj = pca.transform(z.reshape(1, -1))[0]

        prompt = build_prompt(
            text, label, valence, zproj, action, dialogue_history[dlg]
        )
        reply = call_ollama(prompt)

        dialogue_history[dlg].append(f"U: {text}\nA: {reply}")

        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    dlg,
                    utt,
                    text,
                    label,
                    valence,
                    action,
                    float(zproj[0]),
                    float(zproj[1]),
                    reply,
                ]
            )

        processed += 1
        if processed % 100 == 0:
            print(f"[simulate] Processed {processed}")

    print(f"[simulate] Finished. Output → {out_csv}")


# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_csv",
        default="data/processed/meld_text_audio_video_arcface_states.csv",
    )
    parser.add_argument(
        "--pca_path",
        default="models/pca_states_2.pkl",
    )
    parser.add_argument(
        "--out",
        default="simulation_outputs.csv",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    simulate(args.metadata_csv, args.pca_path, args.out, args.max_rows)
