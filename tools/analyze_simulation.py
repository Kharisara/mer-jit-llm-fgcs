# tools/analyze_simulation.py
import pandas as pd

df = pd.read_csv("demo_ready.csv")

print("Rows:", len(df))
print("\nColumns:", df.columns.tolist())

print("\nSource model usage:")
print(df["source_model"].value_counts())

print("\nSafety distribution:")
print(df["reply_safety"].value_counts())

# Supportiveness heuristic (conservative)
keywords = [
    "sorry", "i understand", "that sounds", "that must",
    "try", "consider", "ground", "help", "support"
]

supportive = df["reply_sentences"].astype(str).str.lower().apply(
    lambda s: any(k in s for k in keywords)
)

print("\nSupportive fraction (heuristic):", round(supportive.mean(), 3))

print("\nSample outputs:")
print(df[["text", "reply_sentences", "reply_safety", "source_model"]].head(5).to_string(index=False))
