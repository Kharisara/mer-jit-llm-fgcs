from pathlib import Path

import pandas as pd

from replaybench.workload import MELD_PREQUALIFIED_IDENTITY_SCHEME, load_replay_frame

CSV_PATH = Path("paper_outputs/replay_input_v260.csv")
MANIFEST_PATH = Path("paper_outputs/replay_input_v260_manifest.json")
REQUIRED = [
    CSV_PATH,
    MANIFEST_PATH,
    Path("configs/fgcs_extended_benchmark.yaml"),
    Path("run_fgcs_extended_benchmark.py"),
]

print("Required active files:")
missing = []
for item in REQUIRED:
    exists = item.is_file()
    print(item, "OK" if exists else "MISSING")
    if not exists:
        missing.append(item.as_posix())

if missing:
    raise SystemExit("Missing active files: " + ", ".join(missing))

frame = load_replay_frame(
    CSV_PATH,
    {
        "identity_scheme": MELD_PREQUALIFIED_IDENTITY_SCHEME,
        "require_portable_paths": True,
    },
)
if list(frame.columns) != ["source_record_id", "diagnostic_action"]:
    raise SystemExit(f"Unexpected active columns: {list(frame.columns)}")
if len(frame) != 11_351 or frame["source_record_id"].nunique() != 11_351:
    raise SystemExit("Active replay identity/cardinality contract failed")
if int(pd.to_numeric(frame["diagnostic_action"]).sum()) != 2_609:
    raise SystemExit("Active diagnostic-action count contract failed")

print("\nrows:", len(frame))
print("columns:", list(frame.columns))
print("unique_source_record_id:", frame["source_record_id"].nunique())
print("diagnostic_action_positive:", int(frame["diagnostic_action"].sum()))
print("\n[PASS] Active v2.6.0 input is minimal, split-qualified, and state-independent.")
