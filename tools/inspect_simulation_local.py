"""
Inspect merged local-LLM simulation output.

Looks for:
  simulation_local_merged.csv

Prints:
  - row count
  - action distribution
  - supportive heuristics
  - z-projection stats
  - sample replies
"""

import pandas as pd
import numpy as np

FILE = "simulation_local_merged.csv"

SUPPORTIVE_KEYWORDS = [
    "sorry",
    "i'm sorry",
    "that sounds",
    "i understand",
    "i hear you",
    "it makes sense",
    "that must be",
    "try",
    "consider",
    "grounding",
    "breathe",
]

def main():
    try:
        df = pd.read_csv(FILE)
    except FileNotFoundError:
        raise FileNotFoundError(f"{FILE} not found. Merge chunks first!")

    print(f"Rows: {len(df)}\n")

    # Action distribution
    if "action" in df.columns:
        print("Action distribution:")
        print(df["action"].value_counts(), "\n")

    # Supportive heuristic
    df["lower_reply"] = df["reply"].astype(str).str.lower()
    df["supportive"] = df["lower_reply"].apply(
        lambda s: any(k in s for k in SUPPORTIVE_KEYWORDS)
    )
    print("Supportive fraction (heuristic):", df["supportive"].mean(), "\n")

    # z-projection stats
    if "z_x" in df.columns and "z_y" in df.columns:
        print("Z-proj ranges:")
        print(
            "x min/max/mean:",
            df["z_x"].min(),
            df["z_x"].max(),
            df["z_x"].mean(),
        )
        print(
            "y min/max/mean:",
            df["z_y"].min(),
            df["z_y"].max(),
            df["z_y"].mean(),
            "\n",
        )

    # Sample replies
    print("Sample replies (first 5):\n")
    print(
        df[["text", "action", "reply"]]
        .head(5)
        .to_string(index=False)
    )

    # Count how many rows changed vs original (if applicable)
    if "source_file" in df.columns:
        print("\nSources in merged dataframe:")
        print(df["source_file"].value_counts())


if __name__ == "__main__":
    main()
