import pandas as pd

files = {
    "BC": "eval_bc.csv",
    "PROXY": "eval_proxy.csv",
    "RANDOM": "eval_random.csv",
}

rows = []
for name, f in files.items():
    df = pd.read_csv(f)
    rows.append({
        "Policy": name,
        "Intervention Rate": round(df['action'].mean(), 2),
        "Safety OK Rate": round((df['reply_safety'] == "ok").mean(), 2),
    })

print(pd.DataFrame(rows))
