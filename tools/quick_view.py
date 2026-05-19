import pandas as pd

df = pd.read_csv("simulation_bc_groq_sample.csv")
print("Rows:", len(df))
print("\nAction distribution:\n", df['action'].value_counts())
print("\nSample replies:\n")
print(df[['text','action','reply']].head(10).to_string(index=False))
