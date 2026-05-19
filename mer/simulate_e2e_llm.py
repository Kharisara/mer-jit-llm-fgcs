"""
mer/simulate_e2e_llm.py

Unsafe end-to-end generative baseline (E2E-LLM).

Definition (paper-aligned):
- Language model is invoked at EVERY conversational turn
- Model autonomously decides WHEN and HOW to respond
- No explicit intervention policy
- No non-intervention option
- No invocation gating
- No structured output enforcement
- No fallback or safety constraints

This baseline is intentionally unsafe and is used ONLY
to empirically contrast policy-first architectural control.

Output columns:
dialogue_id, utt_id, text, label, valence, action, llm_reply

Run (from project root):
python -m mer.simulate_e2e_llm
"""

import os
import csv
import argparse
import requests
import pandas as pd

# -------------------------------------------------
# Ollama config
# -------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:1b"

# -------------------------------------------------
# Label → valence (for logging only)
# -------------------------------------------------
LABEL_TO_VALENCE = {
    "angry": -1.0,
    "anger": -1.0,
    "sad": -1.0,
    "sadness": -1.0,
    "fear": -0.7,
    "disgust": -0.7,
    "neutral": 0.0,
    "surprise": 0.0,
    "happy": 1.0,
    "joy": 1.0,
}

# -------------------------------------------------
# Prompt (INTENTIONALLY UNCONTROLLED)
# -------------------------------------------------
def build_prompt(text: str, label: str) -> str:
    return f"""
The user said:
"{text}"

Detected emotion: {label}

Respond empathetically if appropriate.
""".strip()

# -------------------------------------------------
# Ollama call (free-form, unsafe)
# -------------------------------------------------
def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        },
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["response"].strip()

# -------------------------------------------------
# Simulation
# -------------------------------------------------
def simulate(metadata_csv: str, out_csv: str, max_rows: int | None = None):
    df = pd.read_csv(metadata_csv)

    if max_rows is not None:
        df = df.iloc[:max_rows]

    print(f"[E2E] Loaded {len(df)} rows")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dialogue_id",
            "utt_id",
            "text",
            "label",
            "valence",
            "action",
            "llm_reply",
        ])

        for i, row in df.iterrows():
            dlg = int(row.get("Dialogue_ID", -1))
            utt = int(row.get("Utterance_ID", -1))
            text = str(row.get("text", ""))
            label = str(row.get("label", "neutral")).lower()
            valence = LABEL_TO_VALENCE.get(label, 0.0)

            # -----------------------------------------
            # E2E BASELINE: ALWAYS INTERVENE
            # -----------------------------------------
            action = 1  # implicit intervention, no silence

            prompt = build_prompt(text, label)
            reply = call_ollama(prompt)

            writer.writerow([
                dlg,
                utt,
                text,
                label,
                valence,
                action,
                reply,
            ])

            if (i + 1) % 100 == 0 or i == 0:
                print(f"[E2E] Processed {i + 1}/{len(df)}")

    print(f"[E2E] Finished → {out_csv}")

# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_csv",
        default="data/processed/meld_text_audio_video_arcface_states.csv",
    )
    parser.add_argument(
        "--out",
        default="simulation_outputs_e2e_llm.csv",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    simulate(
        metadata_csv=args.metadata_csv,
        out_csv=args.out,
        max_rows=args.max_rows,
    )

if __name__ == "__main__":
    main()
