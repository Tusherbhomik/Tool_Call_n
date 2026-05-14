#!/usr/bin/env python3
"""
fix_dataset.py — Clean a tool-learning JSONL dataset by removing broken samples
and stripping output to the 4 fields required for training.

Filters out samples that fail:
  1. Schema completeness (must have all 4 required fields)
  2. GT tool names not in tools list
  3. GT arguments missing required parameters from tool schema
  4. Bad conversation role sequence (consecutive same-role turns)
  5. Duplicate content fingerprint (last user msg + GT tool names)

Output contains ONLY: trajectory_id, prompt_messages, tools, ground_truth_calls.

Usage:
    python fix_dataset.py [--data PATH] [--out PATH]
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_dataset import (
    check_schema,
    check_gt_tool_alignment,
    check_gt_param_compliance,
    check_role_sequence,
    content_fingerprint,
    _tool_name_map,
)

KEEP_FIELDS = {"trajectory_id", "prompt_messages", "tools", "ground_truth_calls"}


def _load_jsonl(path: str) -> list[dict]:
    samples, errors = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
    if errors:
        print(f"  [WARN] Skipped {errors} JSON parse errors.")
    return samples


def _strip(sample: dict) -> dict:
    """Keep only the 4 training-required fields."""
    return {k: sample[k] for k in KEEP_FIELDS if k in sample}


def clean(data_path: str, out_path: str):
    print(f"\nLoading {data_path} ...")
    samples = _load_jsonl(data_path)
    print(f"Loaded {len(samples)} samples.")

    kept, dropped = [], []
    drop_reasons: dict[str, int] = {}
    seen_fps: set[str] = set()

    for sample in samples:
        tid = sample.get("trajectory_id", "?")
        reasons = []

        # 1. Schema
        if check_schema(sample):
            reasons.append("schema")

        if not reasons:
            tmap = _tool_name_map(sample)

            # 2. GT tool names must be in tools list
            if check_gt_tool_alignment(sample, tmap):
                reasons.append("gt_tool_missing")

            # 3. GT must include all required params
            if check_gt_param_compliance(sample, tmap):
                reasons.append("gt_param")

            # 4. Conversation role sequence
            if check_role_sequence(sample):
                reasons.append("role_sequence")

        # 5. Duplicate — only track fingerprints of samples that passed all other checks.
        # If we tracked dropped samples too, their fingerprints would falsely block
        # later valid samples that happen to share the same (last_user_msg, gt_tools).
        fp = content_fingerprint(sample)
        if not reasons:
            if fp in seen_fps:
                reasons.append("duplicate")
            else:
                seen_fps.add(fp)

        if reasons:
            dropped.append({"id": tid, "reasons": reasons})
            for r in reasons:
                drop_reasons[r] = drop_reasons.get(r, 0) + 1
        else:
            kept.append(_strip(sample))

    print(f"\nResults:")
    print(f"  Kept    : {len(kept)}  (stripped to 4 training fields)")
    print(f"  Dropped : {len(dropped)}")
    if drop_reasons:
        print(f"\n  Drop breakdown:")
        for reason, count in drop_reasons.items():
            print(f"    {reason:<20} {count}")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for s in kept:
            f.write(json.dumps(s) + "\n")
    print(f"\nClean dataset written to {out}  ({len(kept)} samples)")

    log_path = out.parent / (out.stem + "_drop_log.json")
    with open(log_path, "w") as f:
        json.dump({"total_dropped": len(dropped), "dropped": dropped}, f, indent=2)
    print(f"Drop log written to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean a tool-learning JSONL dataset and strip to 4 training fields.")
    parser.add_argument("--data", default="data/dataset_14_may.jsonl")
    parser.add_argument("--out",  default="data/dataset_14_may_clean.jsonl")
    args = parser.parse_args()

    if not Path(args.data).exists():
        print(f"ERROR: not found: {args.data}")
        sys.exit(1)

    clean(args.data, args.out)


if __name__ == "__main__":
    main()
