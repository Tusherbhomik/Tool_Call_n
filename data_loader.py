"""
data_loader.py — Load multi-turn training data and render prompts.

Input:  JSONL file where each line is a sample with at minimum:
          - prompt_messages: list of {role, content} messages
          - tools: OpenAI function-calling schema list
          - ground_truth_calls: list of {name, arguments} (can be empty)
          - invalid_tools: optional list of distractor tool names
          - dependency_value: optional string

Output: HuggingFace Dataset with columns:
          - prompt (str, chat-template-rendered)
          - ground_truth_calls (JSON str)
          - invalid_tools (JSON str)
          - dependency_value (str)

Handles both:
  - Legacy single-turn format (has "prompt" string directly)
  - New multi-turn format (has "prompt_messages" list)
"""
import json
from pathlib import Path
from collections import Counter
from datasets import Dataset

import config


def _load_raw(path: Path) -> list[dict]:
    """Load JSONL, skipping corrupt lines."""
    rows = []
    skipped = 0
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                skipped += 1
                print(f"  [WARN] skipped line {line_no}: parse error col {e.colno}")
    if skipped:
        print(f"  [WARN] {skipped} corrupt lines dropped")
    return rows


def _render_prompt(sample: dict, tokenizer) -> str | None:
    """Render sample's prompt_messages + tools through the chat template."""
    # Legacy single-turn format: pre-rendered prompt string
    if "prompt" in sample and "prompt_messages" not in sample:
        return sample["prompt"]

    msgs = sample.get("prompt_messages")
    if not msgs or not any(m.get("role") == "user" for m in msgs):
        return None

    # Strip trailing assistant messages — prompt must end with user or tool
    # so add_generation_prompt=True correctly primes the next assistant turn.
    while msgs and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]

    if not msgs:
        return None

    tools = sample.get("tools", [])
    try:
        return tokenizer.apply_chat_template(
            msgs,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizer versions don't accept enable_thinking
        try:
            return tokenizer.apply_chat_template(
                msgs,
                tools=tools if tools else None,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return None


def _validate_sample(sample: dict) -> bool:
    """Check minimum required fields are present."""
    return config.REQUIRED_FIELDS.issubset(set(sample.keys())) or "prompt" in sample


def build_dataset(tokenizer, data_path: Path = None) -> Dataset:
    """Load JSONL → validated, templated HuggingFace Dataset."""
    path = data_path or config.DATA_PATH
    print(f"Loading data: {path}")

    rows = _load_raw(path)
    print(f"  raw samples: {len(rows)}")

    # Distribution reporting
    categories = Counter(r.get("category", r.get("source", "unknown")) for r in rows)
    print("  category distribution:")
    for k, v in categories.most_common():
        print(f"    {k:30s}: {v:5d} ({100*v/len(rows):.1f}%)")

    empty_gt_count = sum(1 for r in rows if r.get("ground_truth_calls") == [])
    print(f"  empty-GT samples: {empty_gt_count} ({100*empty_gt_count/len(rows):.1f}%)")

    # Process each sample
    processed = []
    dropped_no_prompt = 0
    dropped_invalid = 0

    for r in rows:
        if not _validate_sample(r):
            dropped_invalid += 1
            continue

        # Handle step-wise indexing for trajectory-split samples (legacy)
        gt_calls = r.get("ground_truth_calls", [])
        step_idx = r.get("step_index")
        if isinstance(step_idx, int) and isinstance(gt_calls, list) \
                and 1 <= step_idx <= len(gt_calls) and gt_calls \
                and isinstance(gt_calls[0], dict):
            # Only slice when gt_calls is a list-of-lists (trajectory format)
            # For new format, gt_calls is already the next-call list for this step.
            pass

        prompt = _render_prompt(r, tokenizer)
        if prompt is None:
            dropped_no_prompt += 1
            continue

        processed.append({
            "prompt":             prompt,
            "ground_truth_calls": json.dumps(gt_calls),
            "invalid_tools":      json.dumps(r.get("invalid_tools", [])),
            "dependency_value":   str(r.get("dependency_value", "") or ""),
        })

    print(f"  processed: {len(processed)}")
    print(f"  dropped (no prompt): {dropped_no_prompt}")
    print(f"  dropped (invalid fields): {dropped_invalid}")

    ds = Dataset.from_list(processed)
    return ds
