# tools/evaluate_simulation.py
import pandas as pd

df = pd.read_csv("eval_ready.csv")

print("Total rows:", len(df))

print("\nAction distribution:")
print(df["action"].value_counts(normalize=True))

print("\nAction by emotion:")
print(pd.crosstab(df["label"], df["action"], normalize="index"))

# strict supportiveness heuristic
strict_kw = [
    "sorry", "i hear you", "that sounds", "difficult",
    "try", "consider", "breath", "ground"
]

df["supportive_strict"] = df["reply_sentences"].astype(str).str.lower().apply(
    lambda s: any(k in s for k in strict_kw)
)

print("\nSupportiveness (strict):", df["supportive_strict"].mean())

# safety
print("\nSafety labels:")
print(df["reply_safety"].value_counts(normalize=True))

# qualitative samples
print("\nSample adaptive interventions:")
print(
    df[df["action"] == 1][["label", "text", "reply_sentences"]]
    .sample(3, random_state=42)
    .to_string(index=False)
)
