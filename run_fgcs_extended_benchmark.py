#!/usr/bin/env python3
"""
FGCS extended benchmark runner for the policy-first MER-JIT-LLM pipeline.

This script runs deterministic offline replay experiments over:
    workloads/fractions x policy modes x seeds x worker counts

Final policy modes for the FGCS extension:
    risk_proxy, bc, bc_live, random, always, never

Expected default count:
    5 workloads x 6 policies x 3 seeds x 4 worker settings = 360 runs

Outputs are written to the configured logging.output_dir:
    - scaling_and_runtime_results.csv
    - stage_latency_summary.csv
    - determinism_hash_results.csv
    - parallel_speedup_results.csv
    - policy_ablation_costs.csv
    - live_bc_predictions.csv            only if bc_live and enabled
    - trace_*.csv                         only if logging.save_traces=true
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import psutil
except ImportError:  # optional dependency
    psutil = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # torch is only required for bc_live
    torch = None
    nn = None


SUPPORTED_POLICY_MODES = {"risk_proxy", "proxy", "bc", "bc_live", "random", "always", "never"}
DEFAULT_POLICY_ORDER = ["risk_proxy", "bc", "bc_live", "random", "always", "never"]


# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return cfg


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def stable_hash_to_float(*parts: Any) -> float:
    """Deterministic pseudo-random number in [0, 1), independent of workers."""
    s = "|".join(str(p) for p in parts)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16**16)


def trace_hash(actions: Sequence[int]) -> str:
    """Hash an action sequence for deterministic replay comparison."""
    s = ",".join(str(int(a)) for a in actions)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sanitize_token(value: Any) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace("\\", "_").replace(" ", "_")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some older PyTorch builds do not expose this flag.
            pass


def get_process_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Path and state loading
# ---------------------------------------------------------------------------


def _deduplicate_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        key = os.path.normcase(str(p))
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def candidate_paths(raw_path: Any, input_csv: str | Path, state_root: str | Path | None = None) -> List[Path]:
    """
    Generate robust candidates for state_path values stored in CSVs.

    Handles:
        - absolute paths
        - paths relative to current working directory
        - paths relative to the input CSV directory
        - paths relative to dataset.state_root
    """
    if raw_path is None or pd.isna(raw_path):
        return []

    raw = str(raw_path).strip().strip('"').strip("'")
    if not raw:
        return []

    raw = raw.replace("\\", os.sep)
    p = Path(raw)

    if p.is_absolute():
        return [p]

    input_parent = Path(input_csv).resolve().parent
    candidates = [Path.cwd() / p, input_parent / p, Path(raw)]

    if state_root:
        root = Path(str(state_root).replace("\\", os.sep))
        if not root.is_absolute():
            root = Path.cwd() / root
        candidates.insert(0, root / p)

    return _deduplicate_paths(candidates)


def resolve_existing_path(raw_path: Any, input_csv: str | Path, state_root: str | Path | None = None) -> Optional[Path]:
    for p in candidate_paths(raw_path, input_csv=input_csv, state_root=state_root):
        if p.exists():
            return p
    return None


def safe_touch_state(
    state_path: Any,
    input_csv: str | Path,
    state_root: str | Path | None = None,
) -> Tuple[bool, int]:
    """
    Touch/load a small part of a state file to measure state-loading cost.

    This keeps non-bc_live policies comparable without deserializing full tensors.
    """
    path = resolve_existing_path(state_path, input_csv=input_csv, state_root=state_root)
    if path is None:
        return False, 0

    try:
        size_bytes = path.stat().st_size
    except OSError:
        size_bytes = 0

    suffix = path.suffix.lower()

    try:
        if suffix == ".npy":
            _ = np.load(path, allow_pickle=False, mmap_mode="r")
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                _ = list(data.keys())
        elif suffix in {".pt", ".pth"} and torch is not None:
            # For non-live policies, only touch bytes rather than loading full torch objects.
            with open(path, "rb") as f:
                _ = f.read(4096)
        else:
            with open(path, "rb") as f:
                _ = f.read(4096)
    except Exception:
        # State file exists but has unusual serialization; do not crash non-live policies.
        return True, int(size_bytes)

    return True, int(size_bytes)


def _extract_array_from_npz(npz_obj: Any) -> np.ndarray:
    preferred_keys = ["state", "embedding", "features", "x", "arr_0"]
    for key in preferred_keys:
        if key in npz_obj:
            return np.asarray(npz_obj[key])
    keys = list(npz_obj.keys())
    if not keys:
        raise ValueError("NPZ file contains no arrays")
    return np.asarray(npz_obj[keys[0]])


def _extract_array_from_torch_object(obj: Any) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is not installed")

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()

    if isinstance(obj, Mapping):
        preferred_keys = ["state", "embedding", "features", "x", "arr_0", "vector"]
        for key in preferred_keys:
            if key in obj:
                value = obj[key]
                if torch.is_tensor(value):
                    return value.detach().cpu().numpy()
                return np.asarray(value)

        # Fall back to the first tensor-like value.
        for value in obj.values():
            if torch.is_tensor(value):
                return value.detach().cpu().numpy()
            if isinstance(value, (list, tuple, np.ndarray)):
                return np.asarray(value)

    raise ValueError("Could not extract a numeric state vector from the torch object")


def load_state_vector(
    state_path: Any,
    input_csv: str | Path,
    state_root: str | Path | None,
    expected_dim: int,
    strict_dim: bool,
    missing_state_policy: str,
) -> Tuple[np.ndarray, bool, int]:
    """
    Load a state vector for live behavioral-cloning inference.

    missing_state_policy:
        error : raise if state file missing
        zeros : use a zero vector when missing/unreadable
    """
    path = resolve_existing_path(state_path, input_csv=input_csv, state_root=state_root)

    if path is None:
        if missing_state_policy == "zeros":
            return np.zeros(expected_dim, dtype=np.float32), False, 0
        raise FileNotFoundError(f"State file not found for state_path={state_path!r}")

    size_bytes = int(path.stat().st_size) if path.exists() else 0
    suffix = path.suffix.lower()

    try:
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                arr = _extract_array_from_npz(data)
        elif suffix in {".pt", ".pth"}:
            if torch is None:
                raise RuntimeError("bc_live requires PyTorch to load .pt/.pth states")
            obj = torch.load(path, map_location="cpu")
            arr = _extract_array_from_torch_object(obj)
        elif suffix in {".json"}:
            with open(path, "r", encoding="utf-8") as f:
                arr = np.asarray(json.load(f))
        else:
            # Last-resort text/CSV vector support.
            arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
    except Exception as exc:
        if missing_state_policy == "zeros":
            return np.zeros(expected_dim, dtype=np.float32), True, size_bytes
        raise RuntimeError(f"Failed to load state vector from {path}: {exc}") from exc

    vector = np.asarray(arr, dtype=np.float32).reshape(-1)

    if vector.size != expected_dim:
        if strict_dim:
            raise ValueError(
                f"State vector dimension mismatch for {path}: expected {expected_dim}, got {vector.size}"
            )
        if vector.size > expected_dim:
            vector = vector[:expected_dim]
        else:
            vector = np.pad(vector, (0, expected_dim - vector.size), mode="constant")

    return vector.astype(np.float32, copy=False), True, size_bytes


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------


class BCPolicy(nn.Module if nn is not None else object):
    """Behavioral cloning MLP: 512 -> 256 -> 256 -> 2 by default."""

    def __init__(self, input_dim: int = 512, hidden_dims: Sequence[int] = (256, 256), output_dim: int = 2):
        if nn is None:
            raise RuntimeError("PyTorch is required for BCPolicy")
        super().__init__()
        layers: List[Any] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Any) -> Any:
        return self.net(x)


def _torch_load(path: str | Path, device: str) -> Any:
    if torch is None:
        raise RuntimeError("bc_live requires PyTorch, but torch is not installed")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if torch is not None and hasattr(checkpoint, "state_dict") and not isinstance(checkpoint, Mapping):
        return checkpoint.state_dict()

    if isinstance(checkpoint, Mapping):
        # Direct state_dict case.
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint

        for key in ["model_state_dict", "policy_state_dict", "state_dict", "model", "policy", "net"]:
            if key in checkpoint:
                value = checkpoint[key]
                if torch is not None and hasattr(value, "state_dict") and not isinstance(value, Mapping):
                    return value.state_dict()
                if isinstance(value, Mapping):
                    return value

    raise ValueError("Could not extract a PyTorch state_dict from the checkpoint")


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Make checkpoint loading tolerant to common wrappers/key names.

    Supports examples such as:
        module.net.0.weight -> net.0.weight
        model.net.0.weight  -> net.0.weight
        0.weight            -> net.0.weight
        fc1/fc2/out         -> net.0/net.2/net.4
    """
    normalized: Dict[str, Any] = {}

    for raw_key, value in state_dict.items():
        key = str(raw_key)

        changed = True
        while changed:
            changed = False
            for prefix in ["module.", "model.", "policy."]:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True

        # Direct Sequential checkpoint: 0.weight, 2.weight, 4.weight.
        if key[:1].isdigit():
            key = f"net.{key}"

        # Common hand-written MLP names.
        replacements = {
            "fc1.": "net.0.",
            "linear1.": "net.0.",
            "layer1.": "net.0.",
            "fc2.": "net.2.",
            "linear2.": "net.2.",
            "layer2.": "net.2.",
            "fc3.": "net.4.",
            "linear3.": "net.4.",
            "out.": "net.4.",
            "output.": "net.4.",
            "head.": "net.4.",
        }
        for old, new in replacements.items():
            if key.startswith(old):
                key = key.replace(old, new, 1)
                break

        normalized[key] = value

    return normalized


def load_bc_policy_model(policy_cfg: Mapping[str, Any]) -> Tuple[Any, str, int, bool, str]:
    """Load the bc_live model and return model plus execution settings."""
    if torch is None:
        raise RuntimeError(
            "bc_live was requested but PyTorch is not installed. Install torch or remove bc_live from policy_modes."
        )

    model_path = policy_cfg.get("bc_model_path", "checkpoints/jitai_policy_bc.pt")
    device = str(policy_cfg.get("bc_live_device", "cpu"))
    state_dim = int(policy_cfg.get("bc_live_state_dim", 512))
    hidden_dims = policy_cfg.get("bc_live_hidden_dims", [256, 256])
    output_dim = int(policy_cfg.get("bc_live_output_dim", 2))
    torch_threads = int(policy_cfg.get("bc_live_torch_threads", 1))

    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    model_path_obj = Path(str(model_path).replace("\\", os.sep))
    if not model_path_obj.exists():
        raise FileNotFoundError(f"bc_live model checkpoint not found: {model_path_obj}")

    model = BCPolicy(input_dim=state_dim, hidden_dims=hidden_dims, output_dim=output_dim).to(device)
    checkpoint = _torch_load(model_path_obj, device=device)
    state_dict = normalize_state_dict_keys(extract_state_dict(checkpoint))

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load bc_live checkpoint strictly into BCPolicy(512->256->256->2). "
            "Check that the checkpoint architecture matches the YAML settings. "
            f"Original error: {exc}"
        ) from exc

    model.eval()
    strict_dim = bool(policy_cfg.get("bc_live_strict_state_dim", True))
    missing_state_policy = str(policy_cfg.get("missing_state_policy", "error")).lower()

    if missing_state_policy not in {"error", "zeros"}:
        raise ValueError("policy.missing_state_policy must be either 'error' or 'zeros'")

    return model, device, state_dim, strict_dim, missing_state_policy


def load_bc_reference_actions(
    bc_action_csv: str | Path,
    df_full: pd.DataFrame,
    action_column: str = "action",
    key_column: str = "utterance_id",
) -> Dict[int, int]:
    """
    Load offline BC actions from a CSV produced by the original policy-first run.

    Primary behavior:
        - If reference CSV length matches input CSV length, align by row_index.
        - Otherwise, align by key_column, usually utterance_id.
    """
    path = Path(str(bc_action_csv).replace("\\", os.sep))
    if not path.exists():
        raise FileNotFoundError(f"BC reference action CSV not found: {path}")

    ref = pd.read_csv(path).reset_index(drop=True)
    if action_column not in ref.columns:
        raise ValueError(f"BC reference CSV {path} has no action column named {action_column!r}")

    if len(ref) == len(df_full):
        return {int(i): int(a) for i, a in enumerate(ref[action_column].astype(int).tolist())}

    if key_column not in ref.columns or key_column not in df_full.columns:
        raise ValueError(
            f"Cannot align BC reference by length or by key column {key_column!r}; "
            f"input rows={len(df_full)}, reference rows={len(ref)}"
        )

    ref_map = dict(zip(ref[key_column].astype(str), ref[action_column].astype(int)))
    out: Dict[int, int] = {}
    missing = 0
    for i, value in enumerate(df_full[key_column].astype(str).tolist()):
        if value in ref_map:
            out[int(i)] = int(ref_map[value])
        else:
            missing += 1

    if missing:
        raise ValueError(f"BC reference CSV missing {missing} actions when aligned by {key_column!r}")

    return out


def normalize_label(value: Any) -> str:
    """Normalize categorical affect/emotion labels for deterministic proxy policies."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def extract_label_from_row(row: Mapping[str, Any]) -> Any:
    """Return the first available MELD-style label/emotion value from a replay row."""
    candidates = [
        "label",
        "emotion",
        "Emotion",
        "emotion_label",
        "Emotion_Label",
        "meld_emotion",
        "MELD_emotion",
        "affective_label",
    ]
    for col in candidates:
        if col in row:
            value = row.get(col, "")
            if normalize_label(value):
                return value
    return ""


def proxy_action(label: Any, negative_labels: set[str]) -> int:
    """Backward-compatible deterministic proxy action."""
    return 1 if normalize_label(label) in negative_labels else 0


def risk_proxy_action(row: Mapping[str, Any], negative_labels: set[str]) -> int:
    """
    Deterministic affective-risk diagnostic proxy.

    This is an action-diverse diagnostic replay policy, not a clinically valid
    intervention policy. It selects intervention only when the replay label is
    one of the configured negative/sensitive labels.
    """
    label = extract_label_from_row(row)
    return proxy_action(label, negative_labels)


def random_action(row: Mapping[str, Any], seed: int, row_index: int, p: float) -> int:
    utterance_id = row.get("utterance_id", row_index)
    r = stable_hash_to_float("random_policy", seed, utterance_id, row_index)
    return 1 if r < p else 0


def base_action_for_row(
    row: Mapping[str, Any],
    row_index: int,
    policy_mode: str,
    negative_labels: set[str],
    seed: int,
    random_p: float,
    bc_actions: Optional[Mapping[int, int]] = None,
    live_actions: Optional[Mapping[int, int]] = None,
) -> int:
    if policy_mode in {"risk_proxy", "proxy"}:
        return risk_proxy_action(row, negative_labels)

    if policy_mode == "random":
        return random_action(row, seed, row_index, random_p)

    if policy_mode == "always":
        return 1

    if policy_mode == "never":
        return 0

    if policy_mode == "bc":
        if bc_actions is None or row_index not in bc_actions:
            raise KeyError(f"Missing BC reference action for row_index={row_index}")
        return int(bc_actions[row_index])

    if policy_mode == "bc_live":
        if live_actions is None or row_index not in live_actions:
            raise KeyError(f"Missing live BC action for row_index={row_index}")
        return int(live_actions[row_index])

    raise ValueError(f"Unknown policy_mode: {policy_mode}")


def maybe_fault_inject(
    action: int,
    row: Mapping[str, Any],
    row_index: int,
    policy_mode: str,
    seed: int,
    fault_cfg: Mapping[str, Any],
) -> Tuple[int, int]:
    """
    Optional deterministic action-flip fault injection.

    This simulates policy-output corruption by flipping selected actions before
    invocation gating. It should be detected by trace-hash mismatch, not by
    the unauthorized-invocation counter.
    """
    enabled = bool(fault_cfg.get("enabled", False))
    flip_p = float(fault_cfg.get("action_flip_probability", 0.0))
    allowed = set(fault_cfg.get("allowed_policy_modes", ["risk_proxy", "proxy", "random", "never"]))

    if not enabled or flip_p <= 0.0 or policy_mode not in allowed:
        return int(action), 0

    utterance_id = row.get("utterance_id", row_index)
    r = stable_hash_to_float("fault_action_flip", seed, policy_mode, utterance_id, row_index)
    if r < flip_p:
        return 1 - int(action), 1
    return int(action), 0


def maybe_force_unauthorized_invocation(
    action: int,
    row: Mapping[str, Any],
    row_index: int,
    policy_mode: str,
    seed: int,
    fault_cfg: Mapping[str, Any],
) -> int:
    """
    Optional deterministic invocation-boundary violation injection.

    This simulates a generator/service call that occurs even though the policy
    action is 0 and the invocation gate therefore did not authorize generation.
    The policy action remains unchanged. The caller records the actual generation
    execution and derives the unauthorized-invocation indicator from the
    authorization--execution contradiction.
    """
    enabled = bool(fault_cfg.get("enabled", False))
    invoke_p = float(fault_cfg.get("unauthorized_invoke_probability", 0.0))
    allowed = set(fault_cfg.get("allowed_policy_modes", ["risk_proxy", "proxy", "random", "never"]))

    if not enabled or invoke_p <= 0.0 or policy_mode not in allowed:
        return 0

    # Only rows with action=0 are eligible, because action=1 is already authorized.
    if int(action) != 0:
        return 0

    utterance_id = row.get("utterance_id", row_index)
    r = stable_hash_to_float("fault_unauthorized_invoke", seed, policy_mode, utterance_id, row_index)
    return 1 if r < invoke_p else 0


# ---------------------------------------------------------------------------
# Live BC batching
# ---------------------------------------------------------------------------


def compute_bc_live_actions(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    seed: int,
) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]], pd.DataFrame]:
    """
    Compute live behavioral-cloning actions from 512-d state vectors.

    Returns:
        actions_by_row: row_index -> action
        metrics_by_row: row_index -> state/policy timing and diagnostics
        prediction_df: per-row BC probabilities/logits for optional audit logging
    """
    if torch is None:
        raise RuntimeError("bc_live requires PyTorch")

    seed_everything(seed)

    dataset_cfg = cfg.get("dataset", {})
    policy_cfg = cfg.get("policy", {})
    input_csv = dataset_cfg["input_csv"]
    state_root = dataset_cfg.get("state_root", None)

    model, device, state_dim, strict_dim, missing_state_policy = load_bc_policy_model(policy_cfg)
    batch_size = int(policy_cfg.get("bc_live_batch_size", 4096))
    if batch_size <= 0:
        raise ValueError("policy.bc_live_batch_size must be positive")

    actions_by_row: Dict[int, int] = {}
    metrics_by_row: Dict[int, Dict[str, Any]] = {}
    prediction_rows: List[Dict[str, Any]] = []

    rows = df.to_dict(orient="records")

    for batch_start in range(0, len(rows), batch_size):
        batch_items = rows[batch_start: batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch_items)))
        vectors: List[np.ndarray] = []
        state_meta: Dict[int, Dict[str, Any]] = {}

        for row_index, row in zip(batch_indices, batch_items):
            s0 = time.perf_counter()
            vector, state_exists, state_size_bytes = load_state_vector(
                row.get("state_path", ""),
                input_csv=input_csv,
                state_root=state_root,
                expected_dim=state_dim,
                strict_dim=strict_dim,
                missing_state_policy=missing_state_policy,
            )
            s1 = time.perf_counter()
            vectors.append(vector)
            state_meta[row_index] = {
                "state_exists": int(state_exists),
                "state_size_bytes": int(state_size_bytes),
                "state_loading_ms": (s1 - s0) * 1000.0,
            }

        x_np = np.stack(vectors, axis=0).astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).to(device)

        p0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            actions = torch.argmax(logits, dim=1)
        p1 = time.perf_counter()

        logits_np = logits.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy().astype(int)
        per_row_inference_ms = ((p1 - p0) * 1000.0) / max(1, len(batch_items))

        for local_i, row_index in enumerate(batch_indices):
            action = int(actions_np[local_i])
            actions_by_row[row_index] = action
            metrics_by_row[row_index] = {
                **state_meta[row_index],
                "policy_inference_ms": per_row_inference_ms,
                "bc_live_logit_0": float(logits_np[local_i, 0]) if logits_np.shape[1] > 0 else np.nan,
                "bc_live_logit_1": float(logits_np[local_i, 1]) if logits_np.shape[1] > 1 else np.nan,
                "bc_live_prob_0": float(probs_np[local_i, 0]) if probs_np.shape[1] > 0 else np.nan,
                "bc_live_prob_1": float(probs_np[local_i, 1]) if probs_np.shape[1] > 1 else np.nan,
            }

            row = batch_items[local_i]
            prediction_rows.append({
                "row_index": row_index,
                "utterance_id": row.get("utterance_id", row_index),
                "label": row.get("label", ""),
                "seed": seed,
                "bc_live_action": action,
                "bc_live_logit_0": metrics_by_row[row_index]["bc_live_logit_0"],
                "bc_live_logit_1": metrics_by_row[row_index]["bc_live_logit_1"],
                "bc_live_prob_0": metrics_by_row[row_index]["bc_live_prob_0"],
                "bc_live_prob_1": metrics_by_row[row_index]["bc_live_prob_1"],
                "state_exists": metrics_by_row[row_index]["state_exists"],
                "state_size_bytes": metrics_by_row[row_index]["state_size_bytes"],
                "state_loading_ms": metrics_by_row[row_index]["state_loading_ms"],
                "policy_inference_ms": per_row_inference_ms,
            })

    return actions_by_row, metrics_by_row, pd.DataFrame(prediction_rows)


# ---------------------------------------------------------------------------
# Replay/generation pipeline
# ---------------------------------------------------------------------------


def no_generation_result() -> Dict[str, Any]:
    """Return the trace payload for a replay point where generation was not executed."""
    return {
        "response_json": "",
        "safety": "not_invoked",
    }


def execute_generation_stub(label: Any) -> Dict[str, Any]:
    """
    Execute the deterministic downstream generation stub.

    Reaching this function is the observed generation event. Authorization is
    evaluated before this call, and the caller derives any violation by comparing
    the observed call event with the preceding authorization decision.
    """
    label_l = str(label).lower()
    if label_l in {"joy", "surprise"}:
        response = {
            "sentences": [
                "That sounds positive.",
                "We can keep the response brief and structured.",
            ],
            "safety": "ok",
        }
    else:
        response = {
            "sentences": [
                "I hear you.",
                "We can pause here for a moment.",
            ],
            "safety": "ok",
        }

    return {
        "response_json": json.dumps(response, ensure_ascii=False),
        "safety": response["safety"],
    }


def process_one_row(
    row_index: int,
    row_dict: Dict[str, Any],
    cfg: Mapping[str, Any],
    policy_mode: str,
    negative_labels: set[str],
    seed: int,
    bc_actions: Optional[Mapping[int, int]] = None,
    live_actions: Optional[Mapping[int, int]] = None,
    live_metrics: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Process one offline replay decision point and record stage timings."""
    dataset_cfg = cfg.get("dataset", {})
    policy_cfg = cfg.get("policy", {})
    fault_cfg = cfg.get("fault_injection", {})
    input_csv = dataset_cfg["input_csv"]
    state_root = dataset_cfg.get("state_root", None)
    random_p = float(policy_cfg.get("random_intervention_probability", 0.5))

    t0 = time.perf_counter()

    # Stage 1: state loading. For bc_live this has already happened during batch inference.
    if policy_mode == "bc_live" and live_metrics is not None and row_index in live_metrics:
        state_exists = int(live_metrics[row_index].get("state_exists", 0))
        state_size_bytes = int(live_metrics[row_index].get("state_size_bytes", 0))
        state_loading_ms = float(live_metrics[row_index].get("state_loading_ms", 0.0))
    else:
        s0 = time.perf_counter()
        state_exists_bool, state_size_bytes = safe_touch_state(
            row_dict.get("state_path", ""), input_csv=input_csv, state_root=state_root
        )
        s1 = time.perf_counter()
        state_exists = int(state_exists_bool)
        state_loading_ms = (s1 - s0) * 1000.0

    # Stage 2: policy inference/action selection.
    if policy_mode == "bc_live" and live_metrics is not None and row_index in live_metrics:
        p0 = time.perf_counter()
        action = base_action_for_row(
            row=row_dict,
            row_index=row_index,
            policy_mode=policy_mode,
            negative_labels=negative_labels,
            seed=seed,
            random_p=random_p,
            bc_actions=bc_actions,
            live_actions=live_actions,
        )
        # Use measured batch inference time, not the dictionary lookup time.
        _ = time.perf_counter() - p0
        policy_inference_ms = float(live_metrics[row_index].get("policy_inference_ms", 0.0))
    else:
        p0 = time.perf_counter()
        action = base_action_for_row(
            row=row_dict,
            row_index=row_index,
            policy_mode=policy_mode,
            negative_labels=negative_labels,
            seed=seed,
            random_p=random_p,
            bc_actions=bc_actions,
            live_actions=live_actions,
        )
        p1 = time.perf_counter()
        policy_inference_ms = (p1 - p0) * 1000.0

    action_before_fault = int(action)
    action, fault_injected = maybe_fault_inject(
        action=action,
        row=row_dict,
        row_index=row_index,
        policy_mode=policy_mode,
        seed=seed,
        fault_cfg=fault_cfg,
    )

    # Stage 3: invocation gating.
    g0 = time.perf_counter()
    authorized_to_generate = int(action) == 1
    g1 = time.perf_counter()

    # Optional invocation-boundary violation fault. This does not change the
    # policy action. It forces generation after the gate when action=0 so the
    # unauthorized-invocation counter can detect the violation.
    unauthorized_invoke_fault = maybe_force_unauthorized_invocation(
        action=action,
        row=row_dict,
        row_index=row_index,
        policy_mode=policy_mode,
        seed=seed,
        fault_cfg=fault_cfg,
    )

    # Stage 4: observed downstream generation execution.
    # The controlled fault can force the call while authorization remains false.
    gen0 = time.perf_counter()
    should_execute_generation = bool(authorized_to_generate or unauthorized_invoke_fault)
    if should_execute_generation:
        generated = execute_generation_stub(row_dict.get("label", ""))
        generation_invoked = 1
    else:
        generated = no_generation_result()
        generation_invoked = 0

    # Derive the violation from the observed execution and prior authorization.
    unauthorized_invocation = int(generation_invoked == 1 and not authorized_to_generate)
    gen1 = time.perf_counter()

    # Stage 5: logging/result construction.
    log0 = time.perf_counter()
    result: Dict[str, Any] = {
        "row_index": row_index,
        "utterance_id": row_dict.get("utterance_id", row_index),
        "label": row_dict.get("label", ""),
        "split": row_dict.get("split", ""),
        "Dialogue_ID": row_dict.get("Dialogue_ID", ""),
        "Utterance_ID": row_dict.get("Utterance_ID", ""),
        "policy_mode": policy_mode,
        "seed": seed,
        "action_before_fault": int(action_before_fault),
        "action": int(action),
        "action_flip_fault_injected": int(fault_injected),
        "unauthorized_invoke_fault_injected": int(unauthorized_invoke_fault),
        "fault_injected": int(fault_injected) + int(unauthorized_invoke_fault),
        "authorized_to_generate": int(authorized_to_generate),
        "generation_invoked": int(generation_invoked),
        "unauthorized_invocation": int(unauthorized_invocation),
        "response_safety": generated["safety"],
        "response_json": generated["response_json"],
        "state_exists": int(state_exists),
        "state_size_bytes": int(state_size_bytes),
        "state_loading_ms": float(state_loading_ms),
        "policy_inference_ms": float(policy_inference_ms),
        "gating_ms": (g1 - g0) * 1000.0,
        "generation_stub_ms": (gen1 - gen0) * 1000.0,
    }

    if policy_mode == "bc_live" and live_metrics is not None and row_index in live_metrics:
        for key in ["bc_live_logit_0", "bc_live_logit_1", "bc_live_prob_0", "bc_live_prob_1"]:
            result[key] = live_metrics[row_index].get(key, np.nan)

    log1 = time.perf_counter()
    t1 = time.perf_counter()

    result["logging_ms"] = (log1 - log0) * 1000.0
    result["total_latency_ms"] = (t1 - t0) * 1000.0

    return result


def run_replay(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    policy_mode: str,
    negative_labels: set[str],
    seed: int,
    workers: int,
    bc_actions: Optional[Mapping[int, int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[pd.DataFrame]]:
    """Run deterministic replay for one workload/policy/seed/worker setting."""
    if policy_mode not in SUPPORTED_POLICY_MODES:
        raise ValueError(f"Unsupported policy_mode={policy_mode!r}. Supported: {sorted(SUPPORTED_POLICY_MODES)}")

    seed_everything(seed)
    rows = df.to_dict(orient="records")

    mem_before = get_process_memory_mb()
    start = time.perf_counter()

    live_actions: Optional[Dict[int, int]] = None
    live_metrics: Optional[Dict[int, Dict[str, Any]]] = None
    live_prediction_df: Optional[pd.DataFrame] = None

    if policy_mode == "bc_live":
        live_actions, live_metrics, live_prediction_df = compute_bc_live_actions(df=df, cfg=cfg, seed=seed)

    if workers <= 1:
        results = [
            process_one_row(
                row_index=i,
                row_dict=row,
                cfg=cfg,
                policy_mode=policy_mode,
                negative_labels=negative_labels,
                seed=seed,
                bc_actions=bc_actions,
                live_actions=live_actions,
                live_metrics=live_metrics,
            )
            for i, row in enumerate(rows)
        ]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(
                    process_one_row,
                    i,
                    row,
                    cfg,
                    policy_mode,
                    negative_labels,
                    seed,
                    bc_actions,
                    live_actions,
                    live_metrics,
                ): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(futures):
                results.append(future.result())

    end = time.perf_counter()
    mem_after = get_process_memory_mb()

    out_df = pd.DataFrame(results).sort_values("row_index").reset_index(drop=True)
    actions = out_df["action"].astype(int).tolist()
    total_runtime = end - start
    decision_points = len(out_df)
    total_latencies = out_df["total_latency_ms"].astype(float).tolist()

    summary: Dict[str, Any] = {
        "policy_mode": policy_mode,
        "seed": int(seed),
        "workers": int(workers),
        "decision_points": int(decision_points),
        "total_runtime_seconds": float(total_runtime),
        "throughput_points_per_second": float(decision_points / total_runtime) if total_runtime > 0 else 0.0,
        "mean_latency_ms": statistics.mean(total_latencies) if total_latencies else 0.0,
        "median_latency_ms": statistics.median(total_latencies) if total_latencies else 0.0,
        "p95_latency_ms": float(np.percentile(total_latencies, 95)) if total_latencies else 0.0,
        "intervention_rate": float(out_df["action"].mean()) if decision_points else 0.0,
        "unauthorized_invocations": int(out_df["unauthorized_invocation"].sum()),
        "fault_injected_count": int(out_df["fault_injected"].sum()),
        "trace_hash": trace_hash(actions),
        "memory_before_mb": mem_before,
        "memory_after_mb": mem_after,
        "memory_delta_mb": (mem_after - mem_before) if mem_before is not None and mem_after is not None else None,
        "state_missing_count": int((out_df["state_exists"].astype(int) == 0).sum()),
    }

    return out_df, summary, live_prediction_df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize_stage_latency(trace_df: pd.DataFrame) -> Dict[str, Any]:
    stage_cols = [
        "state_loading_ms",
        "policy_inference_ms",
        "gating_ms",
        "generation_stub_ms",
        "logging_ms",
        "total_latency_ms",
    ]

    summary: Dict[str, Any] = {}
    for col in stage_cols:
        if col not in trace_df.columns:
            continue
        values = trace_df[col].astype(float).tolist()
        summary[f"{col}_mean"] = statistics.mean(values) if values else 0.0
        summary[f"{col}_median"] = statistics.median(values) if values else 0.0
        summary[f"{col}_p95"] = float(np.percentile(values, 95)) if values else 0.0
        summary[f"{col}_std"] = float(np.std(values)) if values else 0.0
    return summary


def write_empty_safe_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_speedup_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    speedup_rows: List[Dict[str, Any]] = []
    group_cols = ["dataset_fraction", "policy_mode", "seed"]

    for _, group in summary_df.groupby(group_cols):
        group = group.sort_values("workers")
        base = group.iloc[0]
        base_runtime = float(base["total_runtime_seconds"])
        base_throughput = float(base["throughput_points_per_second"])

        for _, row in group.iterrows():
            runtime = float(row["total_runtime_seconds"])
            throughput = float(row["throughput_points_per_second"])
            speedup_rows.append({
                "dataset_fraction": row["dataset_fraction"],
                "workload_name": row.get("workload_name", f"fraction_{row['dataset_fraction']}"),
                "policy_mode": row["policy_mode"],
                "seed": int(row["seed"]),
                "workers": int(row["workers"]),
                "runtime_seconds": runtime,
                "throughput_points_per_second": throughput,
                "speedup_vs_single_worker": base_runtime / runtime if runtime > 0 else 0.0,
                "throughput_gain_vs_single_worker": throughput / base_throughput if base_throughput > 0 else 0.0,
            })

    return pd.DataFrame(speedup_rows)


def build_policy_cost_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    agg_cols = {
        "total_runtime_seconds": "mean",
        "throughput_points_per_second": "mean",
        "mean_latency_ms": "mean",
        "median_latency_ms": "mean",
        "p95_latency_ms": "mean",
        "intervention_rate": "mean",
        "unauthorized_invocations": "mean",
        "fault_injected_count": "mean",
        "memory_delta_mb": "mean",
        "state_missing_count": "mean",
    }
    existing_agg_cols = {k: v for k, v in agg_cols.items() if k in summary_df.columns}
    return (
        summary_df
        .groupby(["dataset_fraction", "workload_name", "policy_mode", "workers"], as_index=False)
        .agg(existing_agg_cols)
    )


def validate_config(cfg: Mapping[str, Any]) -> None:
    dataset_cfg = cfg.get("dataset", {})
    benchmark_cfg = cfg.get("benchmark", {})
    policy_cfg = cfg.get("policy", {})

    required = [
        (dataset_cfg, "dataset.input_csv"),
        (dataset_cfg, "dataset.fractions"),
        (benchmark_cfg, "benchmark.seeds"),
        (benchmark_cfg, "benchmark.workers"),
        (benchmark_cfg, "benchmark.policy_modes"),
        (policy_cfg, "policy.negative_labels"),
    ]
    for section, full_name in required:
        key = full_name.split(".")[-1]
        if key not in section:
            raise ValueError(f"Missing required config key: {full_name}")

    policy_modes = list(benchmark_cfg.get("policy_modes", []))
    unknown = sorted(set(policy_modes) - SUPPORTED_POLICY_MODES)
    if unknown:
        raise ValueError(f"Unknown policy_modes in config: {unknown}")

    input_csv = Path(str(dataset_cfg["input_csv"]).replace("\\", os.sep))
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if "bc" in policy_modes:
        bc_action_csv = policy_cfg.get("bc_action_csv", "paper_outputs/policy_first_outputs_bc.csv")
        if not Path(str(bc_action_csv).replace("\\", os.sep)).exists():
            raise FileNotFoundError(
                f"policy_mode='bc' requires policy.bc_action_csv, but file was not found: {bc_action_csv}"
            )

    if "bc_live" in policy_modes:
        model_path = policy_cfg.get("bc_model_path", "checkpoints/jitai_policy_bc.pt")
        if not Path(str(model_path).replace("\\", os.sep)).exists():
            raise FileNotFoundError(
                f"policy_mode='bc_live' requires policy.bc_model_path, but file was not found: {model_path}"
            )
        if torch is None:
            raise RuntimeError("policy_mode='bc_live' requires PyTorch, but torch is not installed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FGCS extended policy-first benchmark.")
    parser.add_argument(
        "--config",
        default="configs/fgcs_extended_benchmark.yaml",
        help="Path to FGCS benchmark YAML config.",
    )
    parser.add_argument(
        "--no-validate-files",
        action="store_true",
        help="Skip upfront file existence checks; useful only for dry integration work.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.no_validate_files:
        validate_config(cfg)

    dataset_cfg = cfg["dataset"]
    benchmark_cfg = cfg["benchmark"]
    policy_cfg = cfg["policy"]
    logging_cfg = cfg.get("logging", {})

    input_csv = dataset_cfg["input_csv"]
    fractions = [float(x) for x in dataset_cfg["fractions"]]
    seeds = [int(x) for x in benchmark_cfg["seeds"]]
    workers_list = [int(x) for x in benchmark_cfg["workers"]]
    policy_modes = list(benchmark_cfg.get("policy_modes", DEFAULT_POLICY_ORDER))
    output_dir = Path(logging_cfg.get("output_dir", "paper_outputs/fgcs_extended_benchmark"))
    save_traces = bool(logging_cfg.get("save_traces", True))
    save_live_bc_predictions = bool(logging_cfg.get("save_live_bc_predictions", True))
    negative_labels = {normalize_label(x) for x in policy_cfg.get("negative_labels", [])}

    ensure_dir(output_dir)

    df_full = pd.read_csv(input_csv).reset_index(drop=True)
    if df_full.empty:
        raise ValueError(f"Input CSV has no rows: {input_csv}")

    bc_actions: Optional[Dict[int, int]] = None
    if "bc" in policy_modes:
        bc_actions = load_bc_reference_actions(
            bc_action_csv=policy_cfg.get("bc_action_csv", "paper_outputs/policy_first_outputs_bc.csv"),
            df_full=df_full,
            action_column=str(policy_cfg.get("bc_action_column", "action")),
            key_column=str(policy_cfg.get("bc_key_column", "utterance_id")),
        )

    expected_runs = len(fractions) * len(policy_modes) * len(seeds) * len(workers_list)
    print(f"[INFO] Loaded {len(df_full)} decision points from {input_csv}")
    print(f"[INFO] Workloads/fractions: {fractions}")
    print(f"[INFO] Policy modes: {policy_modes}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Worker settings: {workers_list}")
    print(f"[INFO] Expected benchmark runs: {expected_runs}")

    all_summaries: List[Dict[str, Any]] = []
    all_stage_summaries: List[Dict[str, Any]] = []
    determinism_rows: List[Dict[str, Any]] = []
    live_prediction_frames: List[pd.DataFrame] = []

    for fraction in fractions:
        n = max(1, int(len(df_full) * float(fraction)))
        df = df_full.iloc[:n].copy().reset_index(drop=True)
        workload_name = f"fraction_{sanitize_token(fraction)}"

        for policy_mode in policy_modes:
            reference_hash_by_seed: Dict[Tuple[float, str, int], str] = {}
            reference_intervention_rate_by_seed: Dict[Tuple[float, str, int], float] = {}

            for seed in seeds:
                for workers in workers_list:
                    print(
                        f"[INFO] workload={workload_name}, n={n}, "
                        f"policy={policy_mode}, seed={seed}, workers={workers}"
                    )

                    trace_df, summary, live_pred_df = run_replay(
                        df=df,
                        cfg=cfg,
                        policy_mode=policy_mode,
                        negative_labels=negative_labels,
                        seed=seed,
                        workers=workers,
                        bc_actions=bc_actions,
                    )

                    summary.update({
                        "dataset_fraction": fraction,
                        "workload_name": workload_name,
                        "input_csv": input_csv,
                    })
                    all_summaries.append(summary)

                    stage_summary = summarize_stage_latency(trace_df)
                    stage_summary.update({
                        "dataset_fraction": fraction,
                        "workload_name": workload_name,
                        "decision_points": n,
                        "policy_mode": policy_mode,
                        "seed": seed,
                        "workers": workers,
                    })
                    all_stage_summaries.append(stage_summary)

                    key = (fraction, policy_mode, seed)
                    if workers == workers_list[0]:
                        reference_hash_by_seed[key] = summary["trace_hash"]
                        reference_intervention_rate_by_seed[key] = float(summary["intervention_rate"])

                    reference_hash = reference_hash_by_seed.get(key)
                    reference_ir = reference_intervention_rate_by_seed.get(key)
                    hash_match = int(summary["trace_hash"] == reference_hash) if reference_hash is not None else None
                    intervention_rate_delta = (
                        float(summary["intervention_rate"]) - reference_ir if reference_ir is not None else None
                    )

                    determinism_rows.append({
                        "dataset_fraction": fraction,
                        "workload_name": workload_name,
                        "policy_mode": policy_mode,
                        "seed": seed,
                        "workers": workers,
                        "trace_hash": summary["trace_hash"],
                        "reference_hash": reference_hash,
                        "hash_match": hash_match,
                        "intervention_rate": summary["intervention_rate"],
                        "reference_intervention_rate": reference_ir,
                        "intervention_rate_delta": intervention_rate_delta,
                        "unauthorized_invocations": summary["unauthorized_invocations"],
                        "fault_injected_count": summary["fault_injected_count"],
                    })

                    if save_traces:
                        trace_path = (
                            output_dir
                            / f"trace_{workload_name}_policy_{policy_mode}_seed_{seed}_workers_{workers}.csv"
                        )
                        trace_df.to_csv(trace_path, index=False)

                    if save_live_bc_predictions and live_pred_df is not None and not live_pred_df.empty:
                        live_pred_df = live_pred_df.copy()
                        live_pred_df["dataset_fraction"] = fraction
                        live_pred_df["workload_name"] = workload_name
                        live_pred_df["workers"] = workers
                        live_prediction_frames.append(live_pred_df)

    summary_df = pd.DataFrame(all_summaries)
    stage_df = pd.DataFrame(all_stage_summaries)
    determinism_df = pd.DataFrame(determinism_rows)
    speedup_df = build_speedup_summary(summary_df)
    policy_cost_df = build_policy_cost_summary(summary_df)

    summary_path = output_dir / "scaling_and_runtime_results.csv"
    stage_path = output_dir / "stage_latency_summary.csv"
    det_path = output_dir / "determinism_hash_results.csv"
    speedup_path = output_dir / "parallel_speedup_results.csv"
    policy_cost_path = output_dir / "policy_ablation_costs.csv"

    write_empty_safe_csv(summary_df, summary_path)
    write_empty_safe_csv(stage_df, stage_path)
    write_empty_safe_csv(determinism_df, det_path)
    write_empty_safe_csv(speedup_df, speedup_path)
    write_empty_safe_csv(policy_cost_df, policy_cost_path)

    if save_live_bc_predictions and live_prediction_frames:
        live_bc_path = output_dir / "live_bc_predictions.csv"
        pd.concat(live_prediction_frames, ignore_index=True).to_csv(live_bc_path, index=False)
        print(f"[OUT] {live_bc_path}")

    print("[DONE] FGCS extended benchmark complete.")
    print(f"[OUT] {summary_path}")
    print(f"[OUT] {stage_path}")
    print(f"[OUT] {det_path}")
    print(f"[OUT] {speedup_path}")
    print(f"[OUT] {policy_cost_path}")


if __name__ == "__main__":
    main()

