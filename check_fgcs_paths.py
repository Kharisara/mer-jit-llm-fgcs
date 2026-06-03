import os
import pandas as pd
from pathlib import Path

csv_path = "paper_outputs/replay_input_clean.csv"
df = pd.read_csv(csv_path)

print("rows:", len(df))
print("columns:", list(df.columns))

required = [
    "paper_outputs/replay_input_clean.csv",
    "paper_outputs/policy_first_outputs_bc.csv",
    "checkpoints/jitai_policy_bc.pt",
    "configs/fgcs_extended_benchmark.yaml",
    "run_fgcs_extended_benchmark.py",
]

print("\nRequired files:")
for p in required:
    print(p, "OK" if Path(p).exists() else "MISSING")

if "state_path" in df.columns:
    print("\nFirst 10 state paths:")
    print(df["state_path"].head(10).to_string(index=False))

    missing = 0
    checked = 0

    for p in df["state_path"].dropna().head(1000):
        checked += 1
        normalized = str(p).replace("\\", os.sep)
        if not Path(normalized).exists():
            missing += 1

    print(f"\nState path check first 1000: checked={checked}, missing={missing}")
else:
    print("\nNO state_path column found")