#!/usr/bin/env python3
"""
Combine FGCS RQ7 fault-injection summaries and generate a validation-ablation matrix.

Inputs:
  paper_outputs/fgcs_tables_figures/fgcs_table_fault_action_flip_detection_summary.csv
  paper_outputs/fgcs_tables_figures/fgcs_table_fault_unauthorized_invoke_detection_summary.csv
  paper_outputs/fgcs_tables_figures/fgcs_table_fault_trace_corruption_detection_summary.csv

Outputs:
  paper_outputs/fgcs_tables_figures/fgcs_table_rq7_fault_detection_combined.csv
  paper_outputs/fgcs_tables_figures/fgcs_table_validation_ablation_matrix.csv
"""

from pathlib import Path
import pandas as pd


TABLE_DIR = Path("paper_outputs/fgcs_tables_figures")


INPUTS = [
    TABLE_DIR / "fgcs_table_fault_action_flip_detection_summary.csv",
    TABLE_DIR / "fgcs_table_fault_unauthorized_invoke_detection_summary.csv",
    TABLE_DIR / "fgcs_table_fault_trace_corruption_detection_summary.csv",
]


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalize_detection_rate(value):
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        return round(float(value), 4)
    except Exception:
        return value


def make_combined_rq7_summary() -> pd.DataFrame:
    frames = []
    for path in INPUTS:
        df = read_required(path)
        df["source_file"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Remove duplicated clean rows and keep one clean false-positive control.
    clean_rows = combined[combined["fault_mode"].astype(str).eq("clean_replay")].copy()
    fault_rows = combined[~combined["fault_mode"].astype(str).eq("clean_replay")].copy()

    clean_summary = pd.DataFrame(
        [
            {
                "fault_mode": "clean_replay",
                "fault_category": "false-positive control",
                "runs": int(clean_rows["runs"].sum()) if "runs" in clean_rows.columns else 0,
                "faults_or_corruptions_injected_total": 0,
                "detected_runs": 0,
                "detection_rate": "",
                "false_positive_runs": int(clean_rows["false_positive_runs"].sum())
                if "false_positive_runs" in clean_rows.columns
                else 0,
                "max_unauthorized_invocations": 0,
                "detection_channel": "none",
                "paper_interpretation": "Clean replay and clean trace files produced no detected faults.",
            }
        ]
    )

    rows = []

    for _, row in fault_rows.iterrows():
        fault_mode = str(row["fault_mode"])

        if fault_mode == "action_flip_1_percent":
            category = "policy-output corruption"
            injected_col = "faults_injected_total"
            interpretation = (
                "Injected action flips changed the action sequence and were detected "
                "through SHA-256 trace mismatch."
            )
        elif fault_mode == "unauthorized_invoke_1_percent":
            category = "invocation-boundary violation"
            injected_col = "faults_injected_total"
            interpretation = (
                "Forced generator calls after action=0 were detected as unauthorized "
                "invocations at the invocation boundary."
            )
        elif fault_mode == "trace_action_corruption_1_percent":
            category = "post-hoc action-trace corruption"
            injected_col = "corruptions_injected_total"
            interpretation = (
                "Post-hoc action changes in saved traces were detected through "
                "SHA-256 action-trace mismatch."
            )
        elif fault_mode == "drop_trace_rows_1_percent":
            category = "missing trace rows"
            injected_col = "corruptions_injected_total"
            interpretation = (
                "Dropped trace rows were detected through row-count mismatch and "
                "SHA-256 action-trace mismatch."
            )
        elif fault_mode == "duplicate_trace_rows_1_percent":
            category = "duplicated trace rows"
            injected_col = "corruptions_injected_total"
            interpretation = (
                "Duplicated trace rows were detected through row-count mismatch and "
                "SHA-256 action-trace mismatch."
            )
        else:
            category = "other"
            injected_col = (
                "faults_injected_total"
                if "faults_injected_total" in row.index
                else "corruptions_injected_total"
            )
            interpretation = str(row.get("interpretation", ""))

        rows.append(
            {
                "fault_mode": fault_mode,
                "fault_category": category,
                "runs": int(row.get("runs", 0)),
                "faults_or_corruptions_injected_total": int(row.get(injected_col, 0)),
                "detected_runs": int(row.get("detected_runs", 0)),
                "detection_rate": normalize_detection_rate(row.get("detection_rate", "")),
                "false_positive_runs": int(row.get("false_positive_runs", 0)),
                "max_unauthorized_invocations": int(row.get("max_unauthorized_invocations", 0))
                if "max_unauthorized_invocations" in row.index and not pd.isna(row.get("max_unauthorized_invocations"))
                else "",
                "detection_channel": str(row.get("detection_channel", "")),
                "paper_interpretation": interpretation,
            }
        )

    fault_summary = pd.DataFrame(rows)

    preferred_order = [
        "clean_replay",
        "action_flip_1_percent",
        "unauthorized_invoke_1_percent",
        "trace_action_corruption_1_percent",
        "drop_trace_rows_1_percent",
        "duplicate_trace_rows_1_percent",
    ]

    out = pd.concat([clean_summary, fault_summary], ignore_index=True)
    out["order"] = out["fault_mode"].apply(
        lambda x: preferred_order.index(x) if x in preferred_order else 999
    )
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    return out


def make_validation_ablation_matrix() -> pd.DataFrame:
    """
    This matrix summarizes which validation mechanism detects each observed fault class.
    It is based on the empirical detection channels from the RQ7 runs.
    """

    rows = [
        {
            "fault_mode": "action_flip_1_percent",
            "fault_category": "policy-output corruption",
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "No",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "SHA-256 trace mismatch",
        },
        {
            "fault_mode": "unauthorized_invoke_1_percent",
            "fault_category": "invocation-boundary violation",
            "logging_only": "No",
            "hash_only": "No",
            "gate_only": "Yes",
            "row_count_only": "No",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "unauthorized-invocation counter",
        },
        {
            "fault_mode": "trace_action_corruption_1_percent",
            "fault_category": "post-hoc action-trace corruption",
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "No",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "SHA-256 action-trace mismatch",
        },
        {
            "fault_mode": "drop_trace_rows_1_percent",
            "fault_category": "missing trace rows",
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "Yes",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "row-count mismatch + SHA-256 mismatch",
        },
        {
            "fault_mode": "duplicate_trace_rows_1_percent",
            "fault_category": "duplicated trace rows",
            "logging_only": "No",
            "hash_only": "Yes",
            "gate_only": "No",
            "row_count_only": "Yes",
            "hash_plus_gate": "Yes",
            "full_validator": "Yes",
            "primary_detection_channel": "row-count mismatch + SHA-256 mismatch",
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    combined = make_combined_rq7_summary()
    matrix = make_validation_ablation_matrix()

    combined_path = TABLE_DIR / "fgcs_table_rq7_fault_detection_combined.csv"
    matrix_path = TABLE_DIR / "fgcs_table_validation_ablation_matrix.csv"

    combined.to_csv(combined_path, index=False)
    matrix.to_csv(matrix_path, index=False)

    print("[DONE] Combined RQ7 fault-validation tables generated.")
    print(f"[OUT] {combined_path}")
    print(f"[OUT] {matrix_path}")
    print()
    print("=== Combined RQ7 summary ===")
    print(combined.to_string(index=False))
    print()
    print("=== Validation-ablation matrix ===")
    print(matrix.to_string(index=False))


if __name__ == "__main__":
    main()