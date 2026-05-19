# """
# mer/simulate_with_bc.py

# Deterministic offline replay of a policy-first JITAI architecture.

# Paper-aligned guarantees:
# - Intervention timing is decided ONLY by the policy layer
# - Language generation occurs IFF action == 1
# - Explicit non-intervention (silence) is enforced
# - Behavioural cloning is used for auditability only
# - Deployment-time label gating determines execution behaviour
# - No online interaction
# - No exploration
# - No environment dynamics

# Architectural stress test:
# - Random modality dropout applied during replay only
# - No retraining
# - No policy modification
# - Used to validate robustness of execution guarantees
# """

# import os
# import csv
# import json
# import argparse
# import logging
# import random

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn

# # -------------------------------------------------
# # CONFIGURATION
# # -------------------------------------------------

# STRESS_TEST_ENABLED = True        # ← set False to disable
# MODALITY_DROPOUT_PROB = 0.30      # 30% of turns

# # MELD modality layout
# TEXT_START = 0
# TEXT_END = 768

# AUDIO_START = 768
# AUDIO_END = 2350

# VIDEO_START = 2350
# VIDEO_END = 2862

# # -------------------------------------------------
# # Logging
# # -------------------------------------------------

# logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
# logger = logging.getLogger("simulate_with_bc")

# # -------------------------------------------------
# # Label → valence mapping
# # -------------------------------------------------

# LABEL_TO_VALENCE = {
#     "angry": -1.0,
#     "anger": -1.0,
#     "sad": -1.0,
#     "sadness": -1.0,
#     "fear": -0.7,
#     "disgust": -0.7,
#     "neutral": 0.0,
#     "surprise": 0.0,
#     "happy": 1.0,
#     "joy": 1.0,
# }

# NEGATIVE_LABELS = {"angry", "sad", "fear", "disgust"}

# # -------------------------------------------------
# # Behavioural Cloning Policy (architecture only)
# # -------------------------------------------------

# class BCPolicy(nn.Module):
#     def __init__(self, input_dim=512, hidden_dim=256, num_actions=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_actions),
#         )

#     def forward(self, x):
#         return self.net(x)


# # -------------------------------------------------
# # Load BC model (auditability only)
# # -------------------------------------------------

# bc_policy = None
# bc_device = "cpu"
# bc_model_path = "checkpoints/jitai_policy_bc.pt"

# if os.path.exists(bc_model_path):
#     try:
#         bc_policy = BCPolicy().to(bc_device)
#         bc_policy.load_state_dict(
#             torch.load(bc_model_path, map_location=bc_device)
#         )
#         bc_policy.eval()
#         logger.info("Loaded behavioural cloning policy")
#     except Exception as e:
#         logger.warning(f"BC load failed: {e}")
#         bc_policy = None
# else:
#     logger.warning("BC checkpoint not found — proxy execution only")


# # -------------------------------------------------
# # Architectural stress test
# # -------------------------------------------------

# def apply_modality_dropout(state: np.ndarray) -> np.ndarray:
#     """
#     Randomly remove one modality during replay.
#     Used ONLY to test architectural robustness.
#     """
#     if random.random() > MODALITY_DROPOUT_PROB:
#         return state

#     corrupted = state.copy()

#     if random.choice(["audio", "video"]) == "audio":
#         corrupted[AUDIO_START:AUDIO_END] = 0.0
#     else:
#         corrupted[VIDEO_START:VIDEO_END] = 0.0

#     return corrupted


# # -------------------------------------------------
# # Deterministic safe response stub
# # -------------------------------------------------

# def make_stub_reply(valence: float):
#     if valence < 0:
#         return {
#             "sentences": [
#                 "I hear you. That sounds difficult.",
#                 "Try taking a few slow breaths to steady yourself."
#             ],
#             "safety": "ok",
#         }
#     else:
#         return {
#             "sentences": [
#                 "That sounds positive.",
#                 "Consider one small step to build on that."
#             ],
#             "safety": "ok",
#         }


# # -------------------------------------------------
# # Offline simulation
# # -------------------------------------------------

# def simulate(csv_path: str, out_path: str, max_rows: int = 0, policy_mode: str = "bc"):
#     logger.info(f"Loading dataset: {csv_path}")
#     df = pd.read_csv(csv_path)

#     if max_rows > 0:
#         df = df.iloc[:max_rows]

#     out_columns = list(df.columns) + [
#         "action",
#         "policy_source",
#         "reply_json",
#         "reply_sentences",
#         "reply_safety",
#     ]

#     with open(out_path, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=out_columns)
#         writer.writeheader()

#         for i, row in df.reset_index(drop=True).iterrows():

#             label = str(row.get("label", "neutral")).lower()
#             valence = LABEL_TO_VALENCE.get(label, 0.0)

#             # -----------------------------------------
#             # POLICY DECISION
#             # -----------------------------------------

#             if policy_mode == "random":
#                 action = random.choice([0, 1])
#                 policy_source = "RANDOM"

#             elif policy_mode == "proxy":
#                 action = 1 if valence < 0 else 0
#                 policy_source = "PROXY"

#             else:
#                 try:
#                     if bc_policy is None:
#                         raise ValueError("BC unavailable")

#                     state_path = row.get("state_path")
#                     if not isinstance(state_path, str) or not os.path.exists(state_path):
#                         raise ValueError("Missing state file")

#                     state = np.load(state_path).reshape(-1)

#                     if STRESS_TEST_ENABLED:
#                         state = apply_modality_dropout(state)

#                     state_t = (
#                         torch.from_numpy(state.reshape(1, -1))
#                         .float()
#                         .to(bc_device)
#                     )

#                     with torch.no_grad():
#                         _ = bc_policy(state_t)

#                     # FINAL EXECUTION DECISION
#                     # (paper-consistent label gating)
#                     action = 1 if label in NEGATIVE_LABELS else 0
#                     policy_source = "BC_LABEL_GATED"

#                 except Exception:
#                     action = 1 if valence < 0 else 0
#                     policy_source = "PROXY_FALLBACK"

#             # -----------------------------------------
#             # POLICY-FIRST GUARANTEE
#             # -----------------------------------------

#             if action == 1:
#                 reply = make_stub_reply(valence)
#                 reply_json = json.dumps(reply)
#                 reply_sentences = json.dumps(reply["sentences"])
#                 reply_safety = reply["safety"]
#             else:
#                 reply_json = None
#                 reply_sentences = None
#                 reply_safety = None

#             out_row = row.to_dict()
#             out_row.update(
#                 {
#                     "action": action,
#                     "policy_source": policy_source,
#                     "reply_json": reply_json,
#                     "reply_sentences": reply_sentences,
#                     "reply_safety": reply_safety,
#                 }
#             )

#             writer.writerow(out_row)

#             if (i + 1) % 100 == 0 or i == 0:
#                 logger.info(f"Processed {i + 1}/{len(df)}")

#     logger.info(f"Simulation complete → {out_path}")


# # -------------------------------------------------
# # CLI
# # -------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True)
#     parser.add_argument("--out", required=True)
#     parser.add_argument("--max_rows", type=int, default=0)
#     parser.add_argument(
#         "--policy_mode",
#         default="bc",
#         choices=["bc", "proxy", "random"],
#     )

#     args = parser.parse_args()

#     simulate(
#         csv_path=args.csv,
#         out_path=args.out,
#         max_rows=args.max_rows,
#         policy_mode=args.policy_mode,
#     )


# if __name__ == "__main__":
#     main()

"""
mer/simulate_with_bc.py

Deterministic offline replay of a policy-first JITAI architecture.

Paper-aligned guarantees:
- Intervention timing is decided ONLY by the policy layer
- Language generation occurs IFF action == 1
- Explicit non-intervention (silence) is enforced
- Behavioural cloning is used for auditability only
- Deployment-time label gating determines execution behaviour
- No online interaction
- No exploration
- No environment dynamics

Optional experimental controls:
- Representation ablation (text / text_audio / full)
- Architectural stress test via modality dropout
"""

import os
import csv
import json
import argparse
import logging
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# -------------------------------------------------
# Deterministic execution control
# -------------------------------------------------
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# If using CUDA (future-proofing)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Ensure deterministic behaviour in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

# Representation mode:
# Options: "text", "text_audio", "full"
REPRESENTATION_MODE = "full"

# Stress test configuration
STRESS_TEST_ENABLED = False
MODALITY_DROPOUT_PROB = 0.30

# MELD modality layout
TEXT_START = 0
TEXT_END = 768

AUDIO_START = 768
AUDIO_END = 2350

VIDEO_START = 2350
VIDEO_END = 2862

# -------------------------------------------------
# Logging
# -------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("simulate_with_bc")

# -------------------------------------------------
# Label → valence mapping
# -------------------------------------------------

LABEL_TO_VALENCE = {
    "angry": -1.0,
    "anger": -1.0,
    "sad": -1.0,
    "sadness": -1.0,
    "fear": -0.7,
    "disgust": -0.7,
    "neutral": 0.0,
    "surprise": 0.0,
    "happy": 1.0,
    "joy": 1.0,
}

NEGATIVE_LABELS = {"angry", "sad", "fear", "disgust"}

# -------------------------------------------------
# Behavioural Cloning Policy (architecture only)
# -------------------------------------------------

class BCPolicy(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_actions=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Load BC model (auditability only)
# -------------------------------------------------

bc_policy = None
bc_device = "cpu"
bc_model_path = "checkpoints/jitai_policy_bc.pt"

if os.path.exists(bc_model_path):
    try:
        bc_policy = BCPolicy().to(bc_device)
        bc_policy.load_state_dict(
            torch.load(bc_model_path, map_location=bc_device)
        )
        bc_policy.eval()
        logger.info("Loaded behavioural cloning policy")
    except Exception as e:
        logger.warning(f"BC load failed: {e}")
        bc_policy = None
else:
    logger.warning("BC checkpoint not found — proxy execution only")

# -------------------------------------------------
# Architectural stress test
# -------------------------------------------------

def apply_modality_dropout(state: np.ndarray) -> np.ndarray:
    """
    Randomly removes audio or video modality.
    Used ONLY for architectural robustness testing.
    """
    if random.random() > MODALITY_DROPOUT_PROB:
        return state

    corrupted = state.copy()

    if random.choice(["audio", "video"]) == "audio":
        corrupted[AUDIO_START:AUDIO_END] = 0.0
    else:
        corrupted[VIDEO_START:VIDEO_END] = 0.0

    return corrupted

# -------------------------------------------------
# Deterministic safe response stub
# -------------------------------------------------

def make_stub_reply(valence: float):
    if valence < 0:
        return {
            "sentences": [
                "I hear you. That sounds difficult.",
                "Try taking a few slow breaths to steady yourself."
            ],
            "safety": "ok",
        }
    else:
        return {
            "sentences": [
                "That sounds positive.",
                "Consider one small step to build on that."
            ],
            "safety": "ok",
        }

def fallback_reply():
    return {
        "sentences": [
            "I'm here with you.",
            "Let's take a moment to pause and breathe."
        ],
        "safety": "fallback",
    }

# -------------------------------------------------
# Offline simulation
# -------------------------------------------------

def simulate(csv_path: str, out_path: str, max_rows: int = 0, policy_mode: str = "bc"):
    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    if max_rows > 0:
        df = df.iloc[:max_rows]

    out_columns = list(df.columns) + [
        "action",
        "policy_source",
        "reply_json",
        "reply_sentences",
        "reply_safety",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_columns)
        writer.writeheader()

        for i, row in df.reset_index(drop=True).iterrows():

            label = str(row.get("label", "neutral")).lower()
            valence = LABEL_TO_VALENCE.get(label, 0.0)

            # -----------------------------------------
            # POLICY DECISION
            # -----------------------------------------

            if policy_mode == "random":
                action = random.choice([0, 1])
                policy_source = "RANDOM"

            elif policy_mode == "proxy":
                action = 1 if valence < 0 else 0
                policy_source = "PROXY"

            else:  # BC (auditability + label gating)
                try:
                    if bc_policy is None:
                        raise ValueError("BC unavailable")

                    state_path = row.get("state_path")
                    if not isinstance(state_path, str) or not os.path.exists(state_path):
                        raise ValueError("Missing state file")

                    state = np.load(state_path).reshape(-1)

                    # -----------------------------------------
                    # Representation ablation (deterministic)
                    # -----------------------------------------

                    if REPRESENTATION_MODE == "text":
                        state[AUDIO_START:VIDEO_END] = 0.0

                    elif REPRESENTATION_MODE == "text_audio":
                        state[VIDEO_START:VIDEO_END] = 0.0

                    # full → unchanged

                    # -----------------------------------------
                    # Optional stress test
                    # -----------------------------------------

                    if STRESS_TEST_ENABLED:
                        state = apply_modality_dropout(state)

                    state_t = (
                        torch.from_numpy(state.reshape(1, -1))
                        .float()
                        .to(bc_device)
                    )

                    # # Forward pass for auditability only
                    # with torch.no_grad():
                    #     _ = bc_policy(state_t)

                    # # FINAL EXECUTION DECISION
                    # action = 1 if label in NEGATIVE_LABELS else 0
                    policy_source = "BC_LABEL_GATED"
                    # Behavioural cloning policy decision
                    with torch.no_grad():
                        logits = bc_policy(state_t)
                        action = int(torch.argmax(logits, dim=1).item())
                    policy_source = "BC"

                except Exception as e:
                    logger.warning(f"BC replay failed at row {i}: {e}")
                    action = 1 if valence < 0 else 0
                    policy_source = "PROXY_FALLBACK"
                
            # -----------------------------------------
            # POLICY-FIRST GUARANTEE
            # -----------------------------------------

            if action == 1:
                reply = make_stub_reply(valence)
                reply_json = json.dumps(reply)
                reply_sentences = json.dumps(reply["sentences"])
                reply_safety = reply["safety"]
            else:
                reply_json = None
                reply_sentences = None
                reply_safety = None

            out_row = row.to_dict()
            out_row.update(
                {
                    "action": action,
                    "policy_source": policy_source,
                    "reply_json": reply_json,
                    "reply_sentences": reply_sentences,
                    "reply_safety": reply_safety,
                }
            )

            writer.writerow(out_row)

            if (i + 1) % 100 == 0 or i == 0:
                logger.info(f"Processed {i + 1}/{len(df)}")

    logger.info(f"Simulation complete → {out_path}")

# -------------------------------------------------
# CLI
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument(
        "--policy_mode",
        default="bc",
        choices=["bc", "proxy", "random"],
    )

    args = parser.parse_args()

    simulate(
        csv_path=args.csv,
        out_path=args.out,
        max_rows=args.max_rows,
        policy_mode=args.policy_mode,
    )

if __name__ == "__main__":
    main()
