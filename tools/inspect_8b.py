# tools/inspect_8b.py
import pandas as pd
fn = "simulation_bc_groq_8b_sample.csv"
df = pd.read_csv(fn)
print("Rows:", len(df))
print("\nAction distribution:\n", df['action'].value_counts())
print("\nModel used distribution:\n", df.get('model_used', pd.Series()).value_counts())
print("\nSample replies (first 10):\n")
print(df[['text','action','model_used','reply']].head(10).to_string(index=False))
