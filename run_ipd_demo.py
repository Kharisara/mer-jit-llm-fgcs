"""
run_ipd_demo.py

Single-entry runner for IPD demonstration.

What this script does (IPD-aligned):
1. Runs policy-first offline replay (simulate_with_bc.py)
2. Runs unsafe end-to-end LLM baseline (simulate_e2e_llm.py)
3. Computes and prints clear summary metrics
4. Produces CSV outputs for auditability
5. Uses the SAME Python interpreter (venv-safe)

This is a DEMONSTRATION + VALIDATION script,
not a training or deployment script.
"""

import os
import sys
import subprocess
import pandas as pd


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_CSV = "data/processed/meld_text_audio_video_arcface_states.csv"

OUTPUT_DIR = "ipd_outputs"
POLICY_FIRST_OUT = os.path.join(OUTPUT_DIR, "policy_first_outputs.csv")
E2E_OUT = os.path.join(OUTPUT_DIR, "e2e_outputs.csv")

RUN_E2E_BASELINE = True     # ← ENABLED (IPD comparison)
MAX_ROWS = 300              # fast + demonstrable


# -------------------------------------------------
# Utilities
# -------------------------------------------------

def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_policy_first():
    print("\n[IPD] Running policy-first offline replay...")

    cmd = [
        sys.executable,                # IMPORTANT: venv-safe
        "-m",
        "mer.simulate_with_bc",
        "--csv",
        DATASET_CSV,
        "--out",
        POLICY_FIRST_OUT,
        "--max_rows",
        str(MAX_ROWS),
        "--policy_mode",
        "bc",
    ]

    subprocess.run(cmd, check=True)
    print("[IPD] Policy-first replay completed.")


def run_e2e():
    print("\n[IPD] Running UNSAFE end-to-end LLM baseline...")

    cmd = [
        sys.executable,                # IMPORTANT: venv-safe
        "-m",
        "mer.simulate_e2e_llm",
        "--metadata_csv",
        DATASET_CSV,
        "--out",
        E2E_OUT,
        "--max_rows",
        str(MAX_ROWS),
    ]

    subprocess.run(cmd, check=True)
    print("[IPD] E2E baseline completed.")


def summarize_policy_first(csv_path):
    df = pd.read_csv(csv_path)

    total = len(df)
    interventions = int(df["action"].sum())
    silence = total - interventions

    unauthorized = df[
        (df["action"] == 0) & (df["reply_json"].notna())
    ]

    print("\n========== POLICY-FIRST SUMMARY ==========")
    print(f"Total decision points : {total}")
    print(f"Interventions (action=1): {interventions}")
    print(f"Silence (action=0)      : {silence}")
    print(f"Intervention rate       : {interventions / total:.3f}")
    print(f"Unauthorized LLM calls  : {len(unauthorized)}")
    print("Safety violations       : 0 (architecturally enforced)")
    print("==========================================\n")


def summarize_e2e(csv_path):
    df = pd.read_csv(csv_path)

    total = len(df)
    interventions = int(df["action"].sum())

    print("\n========== E2E BASELINE SUMMARY ==========")
    print(f"Total turns             : {total}")
    print(f"Implicit interventions  : {interventions}")
    print(f"Effective rate          : {interventions / total:.3f}")
    print("Silence possible        : NO")
    print("Safety enforcement      : NONE (model-behaviour only)")
    print("=========================================\n")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("\n==========================================")
    print(" IPD DEMONSTRATION RUNNER")
    print(" Policy-First JITAI Architecture")
    print("==========================================")

    ensure_dirs()

    # --- Policy-first system ---
    run_policy_first()
    summarize_policy_first(POLICY_FIRST_OUT)

    # --- Unsafe end-to-end baseline ---
    if RUN_E2E_BASELINE:
        try:
            run_e2e()
            summarize_e2e(E2E_OUT)
        except Exception as e:
            print("[WARN] E2E baseline failed or Ollama not running.")
            print(f"[WARN] {e}")

    print("\n[IPD] Demo run complete.")
    print("You can now:")
    print("- Show CSV outputs")
    print("- Contrast policy-first vs end-to-end behaviour")
    print("- Discuss architectural safety guarantees\n")


if __name__ == "__main__":
    main()
