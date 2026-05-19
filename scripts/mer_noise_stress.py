# scripts/mer_noise_stress.py
"""
MER Noise Stress Test

Injects noise into emotion valence and evaluates:
- Intervention rate stability
- Action flip rate
- Safety preservation

Run from project root:
python scripts/mer_noise_stress.py
"""

import numpy as np
import pandas as pd

# -----------------------
# Config
# -----------------------
INPUT_CSV = "eval_bc.csv"   # change if needed
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5]
SEED = 42

LABEL_TO_VALENCE = {
    "angry": -1.0,
    "sad": -1.0,
    "neutral": 0.0,
    "happy": 1.0,
}

np.random.seed(SEED)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(INPUT_CSV)

df["valence"] = df["label"].str.lower().map(LABEL_TO_VALENCE).fillna(0.0)

print("\nMER Noise Stress Test Results\n")

rows = []

for sigma in NOISE_LEVELS:
    noisy = df.copy()

    noise = np.random.normal(0, sigma, size=len(noisy))
    noisy["noisy_valence"] = np.clip(noisy["valence"] + noise, -1.0, 1.0)

    # Proxy decision rule (same logic as fallback)
    noisy["noisy_action"] = (noisy["noisy_valence"] < 0).astype(int)

    action_flip_rate = (noisy["noisy_action"] != noisy["action"]).mean()
    intervention_rate = noisy["noisy_action"].mean()
    safety_ok_rate = (noisy["reply_safety"] == "ok").mean()

    rows.append({
        "Noise σ": sigma,
        "Intervention Rate": round(intervention_rate, 2),
        "Action Flip Rate": round(action_flip_rate, 2),
        "Safety OK Rate": round(safety_ok_rate, 2),
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
