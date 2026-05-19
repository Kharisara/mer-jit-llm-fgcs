import pandas as pd
import numpy as np

def compute_metrics(file_path):
    df = pd.read_csv(file_path)
    rate = df["action"].mean()

    if rate == 0 or rate == 1:
        entropy = 0.0
    else:
        entropy = -(rate*np.log2(rate) + (1-rate)*np.log2(1-rate))

    return round(rate, 3), round(entropy, 3)


print("=== INTERVENTION RATE & ENTROPY ===")
print("Text Only:", compute_metrics("bc_text.csv"))
print("Text+Audio:", compute_metrics("bc_text_audio.csv"))
print("Full Multimodal:", compute_metrics("bc_full.csv"))

print("\n=== AGREEMENT WITH PROXY ===")
proxy = pd.read_csv("proxy.csv")

for name in ["bc_text.csv", "bc_text_audio.csv", "bc_full.csv"]:
    df = pd.read_csv(name)
    agreement = (df["action"] == proxy["action"]).mean()
    print(name, "agreement:", round(agreement, 3))
