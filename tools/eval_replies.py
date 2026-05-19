# tools/eval_replies.py
import pandas as pd, glob, re
fns = sorted(glob.glob("simulation_bc_groq_8b_chunk*.csv"))
df = pd.concat([pd.read_csv(f) for f in fns], ignore_index=True)
print("Rows:", len(df))
print("Action distribution:\n", df['action'].value_counts())
if 'model_used' in df.columns:
    print("\nModel used:\n", df['model_used'].fillna('').value_counts())

# Supportive heuristic
keywords = ["sorry","i'm sorry","that sounds","i understand","that must be","try","consider","ground","breath","breathe"]
df['supportive'] = df['reply'].astype(str).str.lower().apply(lambda s: any(k in s for k in keywords))
print("\nSupportive fraction:", df['supportive'].mean())

# Concrete-action detection (one-liner heur)
action_phrases = ["try","do","take a","breathe","breath","look around","name 3","grounding","count to"]
df['has_action'] = df['reply'].astype(str).str.lower().apply(lambda s: any(p in s for p in action_phrases))
print("Fraction with a concrete action suggested:", df['has_action'].mean())

# Example failures (no action when action==1)
bad = df[(df['action']==1) & (~df['has_action'])]
print("\nSample action==1 but no action phrase (up to 5):")
print(bad[['text','reply']].head(5).to_string(index=False))
# Save summary
df.to_csv("simulation_bc_groq_8b_merged.csv", index=False)
print("\nMerged CSV saved -> simulation_bc_groq_8b_merged.csv")
