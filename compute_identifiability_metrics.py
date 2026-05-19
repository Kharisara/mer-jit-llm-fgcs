import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# ------------------
# Load CSV
# ------------------
df = pd.read_csv("data/processed/meld_text_audio_video_arcface_states.csv")

print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# ------------------
# Load state vectors
# ------------------
states = []
for p in tqdm(df["state_path"], desc="Loading states"):
    states.append(np.load(p))

X = np.stack(states)
print("State matrix shape:", X.shape)

# ------------------
# Construct actions
# ------------------
NEGATIVE = {"anger", "sadness", "fear", "disgust"}
a = df["label"].apply(lambda e: 1 if e in NEGATIVE else 0).values
print("Action distribution:", np.bincount(a))

# ------------------
# Clustering
# ------------------
k = 50
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(X)
print("Clusters shape:", clusters.shape)

# ------------------
# ASI
# ------------------
asi_count = 0
for c in range(k):
    actions_in_c = a[clusters == c]
    if len(np.unique(actions_in_c)) >= 2:
        asi_count += 1
ASI = asi_count / k
print("ASI:", ASI)

# ------------------
# LDI
# ------------------
ldi_vals = []
for c in range(k):
    actions_in_c = a[clusters == c]
    if len(actions_in_c) == 0:
        continue
    p1 = np.mean(actions_in_c == 1)
    p0 = 1 - p1
    ldi_vals.append(max(p0, p1))
LDI = np.mean(ldi_vals)
print("LDI:", LDI)

# ------------------
# OAI
# ------------------
OAI = 0
print("OAI:", OAI)

# ------------------
# CDS
# ------------------
eps = 0.05
cds_vals = []
for c in range(k):
    actions_in_c = a[clusters == c]
    if len(actions_in_c) == 0:
        continue
    p1 = np.mean(actions_in_c == 1)
    p0 = 1 - p1
    support = int(p0 > eps) + int(p1 > eps)
    cds_vals.append(1 - support / 2)
CDS = np.mean(cds_vals)
print("CDS:", CDS)
