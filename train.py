"""
train.py — Main GRPO training entry point.

Usage:
    python train.py

Expects:
    - data/train_data.jsonl present
    - HF_TOKEN environment variable set (or leave empty for public models)
    - CUDA GPU available

Reads all hyperparameters from config.py.

Reward design follows ToolRL (Qian et al., 2025):
    R_final = R_format + R_correct  ∈ [-3, 4]
"""
import os
import json
import gc
import statistics
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
from trl import GRPOConfig, GRPOTrainer

import config
from data_loader import build_dataset
from reward import toolrl_reward


# ══════════════════════════════════════════════════════════════════════════════
# ENV CHECKS
# ══════════════════════════════════════════════════════════════════════════════
def _preflight():
    print(f"transformers : {transformers.__version__}")
    print(f"torch        : {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if not config.DATA_PATH.is_file():
        raise FileNotFoundError(f"Training data not found: {config.DATA_PATH}")

    config.SAVE_PATH.mkdir(parents=True, exist_ok=True)
    config.CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL + TOKENIZER
# ══════════════════════════════════════════════════════════════════════════════
def _load_model_and_tokenizer():
    if config.HF_TOKEN:
        login(token=config.HF_TOKEN)

    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_ID,
        trust_remote_code=True,
        token=config.HF_TOKEN or None,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if config.USE_FP16 else (torch.bfloat16 if config.USE_BF16 else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        token=config.HF_TOKEN or None,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=config.LORA_TARGET_MODS,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION WRAPPER (TRL interface)
# ══════════════════════════════════════════════════════════════════════════════

# In-process log populated by reward_fn; cleared before each training run.
_reward_log: list[dict] = []


def reward_fn(completions, ground_truth_calls, **kwargs):
    """
    Interface expected by GRPOTrainer.

    Args:
        completions:         list[str] — one completion per rollout.
        ground_truth_calls:  list[list[dict] | str] — ground-truth tool calls
                             per prompt, broadcast across rollouts by TRL.
                             Each entry is either a list of call dicts or a
                             JSON-encoded string of the same.

    Returns:
        list[float] — one scalar reward per completion, in [-3, 4].
    """
    rewards = []
    for i, completion in enumerate(completions):
        gt_raw = (
            ground_truth_calls[i]
            if isinstance(ground_truth_calls, (list, tuple))
            else ground_truth_calls
        )
        gt = json.loads(gt_raw) if isinstance(gt_raw, str) else gt_raw

        # Normalise arguments/parameters key difference across samples.
        normalised_gt = [
            {
                "name": call.get("name", ""),
                "arguments": call.get("arguments") or call.get("parameters") or {},
            }
            for call in (gt or [])
        ]

        r_fmt = _format_reward(completion, normalised_gt)
        r_cor = _correctness_reward(completion, normalised_gt)
        total = r_fmt + r_cor

        _reward_log.append({
            "r_format":  r_fmt,
            "r_correct": r_cor,
            "total":     total,
        })
        rewards.append(total)

    return rewards


# Import the two sub-components directly so reward_fn can log them separately.
from reward import format_reward as _format_reward
from reward import correctness_reward as _correctness_reward


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ══════════════════════════════════════════════════════════════════════════════
def _build_grpo_config(n_samples: int) -> GRPOConfig:
    effective_batch = config.PER_DEVICE_BATCH * config.GRAD_ACCUM_STEPS

    if config.MAX_STEPS > 0:
        max_steps = config.MAX_STEPS
    else:
        max_steps = (n_samples * config.NUM_EPOCHS) // effective_batch
        max_steps = max(max_steps, 10)

    print(f"  effective batch : {effective_batch}")
    print(f"  max_steps       : {max_steps}")
    print(f"  num_generations : {config.NUM_GENERATIONS}")
    print(f"  epochs target   : {config.NUM_EPOCHS}")

    return GRPOConfig(
        output_dir                  = str(config.CKPT_DIR),
        num_train_epochs            = config.NUM_EPOCHS,
        per_device_train_batch_size = config.PER_DEVICE_BATCH,
        gradient_accumulation_steps = config.GRAD_ACCUM_STEPS,
        num_generations             = config.NUM_GENERATIONS,
        max_completion_length       = config.MAX_COMPLETION_LEN,
        learning_rate               = config.LEARNING_RATE,
        beta                        = config.BETA,
        bf16                        = config.USE_BF16,
        fp16                        = config.USE_FP16,
        logging_steps               = config.LOGGING_STEPS,
        save_steps                  = config.SAVE_STEPS,
        max_steps                   = max_steps,
        temperature                 = config.TEMPERATURE,
        seed                        = config.SEED,
        report_to                   = config.REPORT_TO,
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST-TRAINING REPORT
# ══════════════════════════════════════════════════════════════════════════════
def _print_training_report(log: list[dict]):
    if not log:
        print("[WARN] no reward log entries to report")
        return

    n = len(log)
    print(f"\n{'='*60}")
    print(f"ToolRL Reward Stats ({n} evaluations):")
    print('='*60)

    for k in ("r_format", "r_correct", "total"):
        vals = [r[k] for r in log]
        print(f"  {k:12s}: mean={statistics.mean(vals):+.3f}  "
              f"min={min(vals):+.3f}  max={max(vals):+.3f}")

    # Format pass rate (R_format == 1)
    fmt_rate = sum(1 for r in log if r["r_format"] == 1.0) / n
    print(f"\n  Format pass rate : {fmt_rate:.1%}")

    # Quartile trend (meaningful only when we have enough data)
    if n > 200:
        q = n // 4
        print(f"\n  Quartile trend (first → last {q} samples):")
        for k in ("r_format", "r_correct", "total"):
            first = statistics.mean(r[k] for r in log[:q])
            last  = statistics.mean(r[k] for r in log[-q:])
            delta = last - first
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
            print(f"    {k:12s}: {first:+.3f} → {last:+.3f}  Δ={delta:+.3f} {arrow}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MERGED MODEL
# ══════════════════════════════════════════════════════════════════════════════
def _save_merged(model, tokenizer, trainer):
    print("\nMerging LoRA weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(str(config.SAVE_PATH), safe_serialization=True)
    tokenizer.save_pretrained(str(config.SAVE_PATH))

    # Patch config for V100 fp16 inference
    cfg_path = config.SAVE_PATH / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if config.USE_FP16 and cfg.get("torch_dtype") != "float16":
        cfg["torch_dtype"] = "float16"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("  patched torch_dtype → float16")

    del model, merged, trainer
    gc.collect()
    torch.cuda.empty_cache()

    total_mb = sum(
        os.path.getsize(config.SAVE_PATH / fn) / 1024 / 1024
        for fn in os.listdir(config.SAVE_PATH)
    )
    print(f"  saved to   : {config.SAVE_PATH}")
    print(f"  total size : {total_mb:.1f} MB")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _preflight()
    model, tokenizer = _load_model_and_tokenizer()
    dataset = build_dataset(tokenizer)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after filtering.")

    grpo_cfg = _build_grpo_config(n_samples=len(dataset))

    trainer = GRPOTrainer(
        model            = model,
        processing_class = tokenizer,
        reward_funcs     = reward_fn,
        args             = grpo_cfg,
        train_dataset    = dataset,
    )

    _reward_log.clear()
    print("\nStarting GRPO training...")
    print("-" * 60)
    stats = trainer.train()
    print("-" * 60)
    print(f"\nDone. steps={stats.global_step}  "
          f"final_loss={stats.metrics.get('train_loss', 'N/A')}")

    _print_training_report(_reward_log)
    _save_merged(model, tokenizer, trainer)


if __name__ == "__main__":
    main()