# tools/inspect_simulation_bc.py
import pandas as pd
import numpy as np

fn = "simulation_outputs_bc.csv"
df = pd.read_csv(fn)

print("Rows:", len(df))
print("\nAction distribution:")
print(df['action'].value_counts())

# supportive heuristic
keywords = ["sorry","i'm sorry","that sounds","i understand","that must be","it makes sense","try","consider","grounding"]
df['supportive'] = df['reply'].astype(str).str.lower().apply(lambda s: any(k in s for k in keywords))
print("\nSupportive fraction (heuristic):", df['supportive'].mean())

print("\nZ-proj ranges:")
print("x min/max/mean:", df['z_x'].min(), df['z_x'].max(), df['z_x'].mean())
print("y min/max/mean:", df['z_y'].min(), df['z_y'].max(), df['z_y'].mean())

print("\nSample replies where action==1 (3):")
print(df[df['action']==1][['text','reply']].head(3).to_string(index=False))

print("\nSample replies where action==0 (3):")
print(df[df['action']==0][['text','reply']].head(3).to_string(index=False))

# quick diff vs previous simulation if exists
import os
if os.path.exists("simulation_outputs.csv"):
    a = pd.read_csv("simulation_outputs.csv")
    b = df
    diff = (a['reply'] != b['reply']).sum() if len(a) == len(b) else None
    print("\nReplies changed vs simulation_outputs.csv:", diff)
