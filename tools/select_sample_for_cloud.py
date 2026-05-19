# tools/select_sample_for_cloud.py
import pandas as pd
import numpy as np
import os

# Settings
MERGED_CSV = "simulation_bc_groq_8b_merged.csv"  # change if your merged file name differs
OUT_SAMPLE = "sample_for_cloud.csv"
TARGET_FRACTION = 0.08  # ~8%
RANDOM_SEED = 42

def load_df():
    if not os.path.exists(MERGED_CSV):
        raise FileNotFoundError(f"{MERGED_CSV} not found. Run merge first.")
    return pd.read_csv(MERGED_CSV)

def stratified_valence_sample(df, frac=0.08, seed=42):
    np.random.seed(seed)
    # Create a bucket for valence and label to ensure coverage
    df['val_bucket'] = pd.cut(df['valence'].fillna(0.0), bins=[-2,-0.5,0.0,0.5,2], labels=False)
    df['label_cat'] = df['label'].fillna('neutral')
    # Group by label + val_bucket and sample proportionally
    groups = df.groupby(['label_cat','val_bucket'])
    samples = []
    for name, grp in groups:
        k = max(1, int(len(df) * frac * (len(grp) / len(df))))
        k = min(len(grp), k)
        if k <= 0:
            continue
        samp = grp.sample(n=k, random_state=seed)
        samples.append(samp)
    sample_df = pd.concat(samples).drop(columns=['val_bucket','label_cat'])
    # If sample too small or large, adjust by random sampling
    target_n = max(1, int(len(df) * frac))
    if len(sample_df) > target_n:
        sample_df = sample_df.sample(n=target_n, random_state=seed)
    elif len(sample_df) < target_n:
        remaining = df.drop(sample_df.index)
        add = remaining.sample(n=(target_n - len(sample_df)), random_state=seed)
        sample_df = pd.concat([sample_df, add])
    return sample_df

def main():
    df = load_df()
    sample_df = stratified_valence_sample(df, frac=TARGET_FRACTION, seed=RANDOM_SEED)
    sample_df.to_csv(OUT_SAMPLE, index=False)
    print(f"Saved sample {OUT_SAMPLE} rows={len(sample_df)}")
    # brief summary
    print(sample_df['label'].value_counts())
    print("Valence stats:", sample_df['valence'].describe())

if __name__ == "__main__":
    main()
