#!/usr/bin/env python3
"""
Clean streamed-local-llm replies saved as many JSON objects per cell.
Usage:
  python tools/clean_local_replies.py --in simulation_local_merged.csv --out simulation_local_merged_clean.csv
"""

import argparse
import json
import pandas as pd
import sys

def extract_from_stream_text(s: str) -> str:
    """
    s is often newline-separated JSON fragments like:
      {"model":"gemma3:1b","created_at":"...","response":"Here", "done":false}
      {"model":"gemma3:1b","created_at":"...","response":" are", "done":false}
      ...
      {"model":"gemma3:1b", ... "response":"", "done":true}
    This function aggregates the "response" fields in order and returns cleaned text.
    Falls back to returning original string if parsing fails.
    """
    if not isinstance(s, str) or len(s.strip()) == 0:
        return ""
    parts = []
    had_json = False
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "response" in obj:
                parts.append(obj.get("response") or "")
                had_json = True
            else:
                # if top-level is not dict or no response field, try to salvage by ignoring
                continue
        except Exception:
            # not valid json line — some systems prefix streams or include garbage, try to strip braces
            # fallback: continue (we won't raise)
            continue
    if had_json:
        # join and normalize whitespace
        text = "".join(parts)
        # collapse multiple spaces/newlines
        text = " ".join(text.split())
        return text.strip()
    # fallback heuristics:
    # sometimes replies are a Python str repr or already plain text
    return s.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    args = parser.parse_args()

    print("Loading:", args.infile)
    df = pd.read_csv(args.infile, dtype=str).fillna("")

    # new column
    cleaned = []
    for i, raw in enumerate(df["reply"].astype(str)):
        clean = extract_from_stream_text(raw)
        cleaned.append(clean)
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(df)}")

    df["reply_clean"] = cleaned
    df.to_csv(args.outfile, index=False)
    print("Wrote cleaned file:", args.outfile)

if __name__ == "__main__":
    main()
