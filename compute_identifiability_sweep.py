import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

# ------------------
# Load data
# ------------------
df = pd.read_csv("data/processed/meld_text_audio_video_arcface_states.csv")

# Load states
states = []
for p in tqdm(df["state_path"], desc="Loading states"):
    states.append(np.load(p))
X = np.stack(states)

# Cluster once (IMPORTANT: fixed across LPs)
k = 50
kmeans = KMeans(n_clusters=k, random_state=0)
clusters = kmeans.fit_predict(X)

# ------------------
# Logging policy generator
# ------------------
NEGATIVE = {"anger", "sadness", "fear", "disgust"}

def generate_actions(policy_id, seed=0):
    rng = np.random.default_rng(seed)
    actions = []

    for label in df["label"]:
        if policy_id == "LP1":  # deterministic (baseline)
            a = 1 if label in NEGATIVE else 0

        elif policy_id == "LP2":  # stochastic negative
            if label in NEGATIVE:
                a = rng.random() < 0.7
            else:
                a = 0

        elif policy_id == "LP3":  # negative + neutral
            if label in NEGATIVE:
                a = 1
            elif label == "neutral":
                a = rng.random() < 0.3
            else:
                a = 0

        elif policy_id == "LP4":  # high-frequency
            a = rng.random() < 0.5

        elif policy_id == "LP5":  # balanced stress-test
            a = rng.random() < 0.5

        actions.append(int(a))

    return np.array(actions)

# ------------------
# Audit metrics
# ------------------
def compute_audit_metrics(actions):
    # ASI
    asi_count = 0
    for c in range(k):
        a_c = actions[clusters == c]
        if len(np.unique(a_c)) >= 2:
            asi_count += 1
    ASI = asi_count / k

    # LDI
    ldi_vals = []
    for c in range(k):
        a_c = actions[clusters == c]
        if len(a_c) == 0:
            continue
        p1 = np.mean(a_c == 1)
        p0 = 1 - p1
        ldi_vals.append(max(p0, p1))
    LDI = np.mean(ldi_vals)

    # CDS
    eps = 0.05
    cds_vals = []
    for c in range(k):
        a_c = actions[clusters == c]
        if len(a_c) == 0:
            continue
        p1 = np.mean(a_c == 1)
        p0 = 1 - p1
        support = int(p0 > eps) + int(p1 > eps)
        cds_vals.append(1 - support / 2)
    CDS = np.mean(cds_vals)

    return ASI, LDI, CDS

# ------------------
# Run sweep
# ------------------
policies = ["LP1", "LP2", "LP3", "LP4", "LP5"]

print("\nLogging-Policy Sweep (k = 50)")
print("-" * 60)
print(f"{'Policy':<6} | {'ASI':>5} | {'LDI':>5} | {'CDS':>5}")
print("-" * 60)

for pid in policies:
    a = generate_actions(pid)
    ASI, LDI, CDS = compute_audit_metrics(a)
    print(f"{pid:<6} | {ASI:5.2f} | {LDI:5.2f} | {CDS:5.2f}")
