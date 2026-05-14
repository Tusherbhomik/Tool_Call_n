"""
config.py — All training hyperparameters and paths.

Edit this file to change run configuration. Everything else imports from here.
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/mnt/c/Tools/project")
DATA_PATH = PROJECT_ROOT / "data" / "dataset_14_may.jsonl"
SAVE_PATH    = PROJECT_ROOT / "runs" / "qwen2.5-1.5b-2627"
CKPT_DIR     = PROJECT_ROOT / "runs" / "qwen2.5-1.5b-2627-ckpt"

# ── Model ─────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
HF_TOKEN      = ""

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R           = 32
LORA_ALPHA       = 64
LORA_DROPOUT     = 0.05
LORA_TARGET_MODS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── GRPO ──────────────────────────────────────────────────────────────────────
# Effective batch = PER_DEVICE_BATCH × GRAD_ACCUM_STEPS = 2 × 16 = 32
PER_DEVICE_BATCH     = 2
GRAD_ACCUM_STEPS     = 16
NUM_GENERATIONS      = 8         # rollouts per prompt (GRPO group size)
NUM_EPOCHS           = 4
LEARNING_RATE        = 1e-6      # ToolRL-aligned
BETA                 = 0.0       # KL disabled (ToolRL-aligned)
TEMPERATURE          = 1.0
MAX_COMPLETION_LEN   = 512
MAX_STEPS            = -1       # -1 = compute from epochs; set int to override
SEED                 = 42

# ── Precision ─────────────────────────────────────────────────────────────────
USE_FP16 = True   # V100 requires fp16 (no native bf16 support)
USE_BF16 = False

# ── Reward function thresholds ────────────────────────────────────────────────
# Smooth-gate reward config (see reward.py)
REWARD_FORMAT_FLOOR  = 0.0
REWARD_CORRECT_MIN   = -3.0
REWARD_CORRECT_MAX   = 3.0
REWARD_QUALITY_MIN   = -0.5
REWARD_QUALITY_MAX   = 1.0

# ── Logging ───────────────────────────────────────────────────────────────────
LOGGING_STEPS = 5
SAVE_STEPS    = 50
REPORT_TO     = "none"   # set to "wandb" or "tensorboard" if wanted

# ── Dataset expected fields ───────────────────────────────────────────────────
# Used by data_loader.py for validation. Samples missing required fields are dropped.
REQUIRED_FIELDS = {
    "trajectory_id",
    "prompt_messages",
    "tools",
    "ground_truth_calls",
}
