"""
Merge all local-LLM simulation chunks into one CSV.

Looks for:
  simulation_local_chunk01.csv
  simulation_local_chunk02.csv
  simulation_local_chunk03.csv
  simulation_local_chunk04.csv
  simulation_local_chunk05.csv
  simulation_local_chunk06.csv

Outputs:
  simulation_local_merged.csv
"""

import os
import pandas as pd
import glob

CHUNK_PATTERN = "simulation_local_chunk*.csv"
OUT_FILE = "simulation_local_merged.csv"

def main():
    files = sorted(glob.glob(CHUNK_PATTERN))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {CHUNK_PATTERN}")

    print("Found chunk files:")
    for f in files:
        print("  -", f)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)

    merged.to_csv(OUT_FILE, index=False)
    print(f"\nMerged {len(files)} chunks → {OUT_FILE}")
    print(f"Total rows: {len(merged)}")

if __name__ == "__main__":
    main()
