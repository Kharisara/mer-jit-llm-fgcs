import pandas as pd
import glob
import os

# Folder where your chunk CSVs live
PATTERN = "simulation_bc_groq_8b_chunk*.csv"  # adjust if needed
OUT = "simulation_bc_groq_8b_merged.csv"

def main():
    files = sorted(glob.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {PATTERN}")

    print("Found chunk files:")
    for f in files:
        print("  -", f)

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(OUT, index=False)

    print(f"\nMerged {len(files)} files → {OUT}")
    print("Total rows:", len(merged))

if __name__ == "__main__":
    main()
