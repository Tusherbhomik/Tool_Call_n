#!/usr/bin/env python3
"""
audit_dataset.py — Data quality audit for tool-learning JSONL datasets.

Runs 8 checks and writes a quality_report.json with full issue details
and distribution statistics.

Usage:
    python audit_dataset.py [--data PATH] [--out PATH]
"""

import json
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from reward import toolrl_reward
    _REWARD_AVAILABLE = True
except Exception as e:
    print(f"[WARN] reward.py not importable ({e}) — reward sanity check skipped.")
    _REWARD_AVAILABLE = False

REQUIRED_FIELDS = {"trajectory_id", "prompt_messages", "tools", "ground_truth_calls"}


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> tuple[list[dict], list[str]]:
    samples, parse_errors = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                parse_errors.append(f"line {i+1}: {e}")
    return samples, parse_errors


# ─────────────────────────────────────────────────────────────────────────────
# Individual checks
# ─────────────────────────────────────────────────────────────────────────────

def _tool_name_map(sample: dict) -> dict[str, dict]:
    """Map tool name → its parameters schema."""
    out = {}
    for tool in sample.get("tools", []):
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if name:
            out[name] = fn.get("parameters", {})
    return out


def check_schema(sample: dict) -> list[str]:
    return [f for f in REQUIRED_FIELDS if f not in sample]


def check_gt_tool_alignment(sample: dict, tool_map: dict) -> list[str]:
    """GT call names that don't exist in the tools list."""
    return [
        c.get("name", "")
        for c in sample.get("ground_truth_calls", [])
        if c.get("name", "") and c["name"] not in tool_map
    ]


def check_gt_param_compliance(sample: dict, tool_map: dict) -> list[str]:
    """
    GT calls where required parameters (per the tool's JSON Schema) are absent
    from the call's arguments dict.
    """
    issues = []
    for call in sample.get("ground_truth_calls", []):
        name = call.get("name", "")
        args = call.get("arguments") or call.get("parameters") or {}
        if not isinstance(args, dict):
            args = {}
        schema = tool_map.get(name, {})
        for param in schema.get("required", []):
            if param not in args:
                issues.append(f"{name}.{param}")
    return issues


def check_invalid_tool_overlap(sample: dict) -> list[str]:
    """invalid_tools entries that are also in ground_truth_calls."""
    gt_names = {c.get("name", "") for c in sample.get("ground_truth_calls", [])}
    invalid = set(sample.get("invalid_tools", []))
    return sorted(gt_names & invalid)


def check_role_sequence(sample: dict) -> list[str]:
    """
    Flags role-sequence violations:
    - Non-system first message
    - Consecutive user messages
    - Consecutive assistant messages (without a tool turn between them)
    - System message appearing after index 0
    """
    messages = sample.get("prompt_messages", [])
    issues = []
    if not messages:
        return ["empty prompt_messages"]
    if messages[0].get("role") != "system":
        issues.append("first message is not system")
    for i in range(1, len(messages)):
        prev = messages[i - 1].get("role", "")
        curr = messages[i].get("role", "")
        if prev == "user" and curr == "user":
            issues.append(f"consecutive user turns at index {i}")
        if prev == "assistant" and curr == "assistant":
            issues.append(f"consecutive assistant turns at index {i}")
        if i > 0 and curr == "system":
            issues.append(f"system message at index {i}")
    return issues


def check_reward_sanity(sample: dict) -> float | None:
    """
    Builds a perfect mock completion from the GT and runs toolrl_reward.
    A valid GT should always score ≥ 3.0 (format=1 + correctness=+3).
    Returns None if reward module unavailable.
    """
    if not _REWARD_AVAILABLE:
        return None

    gt = sample.get("ground_truth_calls", [])
    normalised_gt = [
        {"name": c.get("name", ""), "arguments": c.get("arguments") or c.get("parameters") or {}}
        for c in gt
    ]

    if normalised_gt:
        calls_str = "".join(
            f'<tool_call>{json.dumps(c)}</tool_call>' for c in normalised_gt
        )
        completion = f"<think>Calling required tools.</think>{calls_str}"
    else:
        completion = "<think>No tool needed for this request.</think><response>Acknowledged.</response>"

    return toolrl_reward(completion, normalised_gt)


def content_fingerprint(sample: dict) -> str:
    """Fingerprint = (last user message text, sorted GT tool names)."""
    last_user = ""
    for m in reversed(sample.get("prompt_messages", [])):
        if m.get("role") == "user":
            last_user = m.get("content", "").strip()
            break
    gt_names = ",".join(sorted(c.get("name", "") for c in sample.get("ground_truth_calls", [])))
    return f"{last_user}||{gt_names}"


# ─────────────────────────────────────────────────────────────────────────────
# Main audit
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(data_path: str) -> dict:
    print(f"\nLoading {data_path} ...")
    samples, parse_errors = load_jsonl(data_path)
    print(f"Loaded {len(samples)} samples  ({len(parse_errors)} JSON parse errors)")

    issues = {
        "json_parse_errors":         parse_errors,
        "schema_missing_field":      [],
        "gt_tool_not_found":         [],
        "gt_missing_required_param": [],
        "bad_role_sequence":         [],
        "duplicates":                [],
        "gt_negative_reward":        [],
    }

    dist = {
        "by_gt_call_count": {},
        "by_message_count": {},
        # informational only — present when samples carry these optional fields
        "by_category":      {},
        "by_domain":        {},
    }

    seen_ids: dict[str, int] = {}
    seen_fps: dict[str, str] = {}
    reward_values: list[float] = []

    for idx, sample in enumerate(samples):
        tid = sample.get("trajectory_id", f"__unknown_{idx}")

        # 1. Schema
        missing = check_schema(sample)
        if missing:
            issues["schema_missing_field"].append({"id": tid, "missing": missing})
            continue  # can't run further checks without required fields

        tool_map = _tool_name_map(sample)

        # 2. GT-tool name alignment
        bad_names = check_gt_tool_alignment(sample, tool_map)
        if bad_names:
            issues["gt_tool_not_found"].append({"id": tid, "unknown_names": bad_names})

        # 3. GT parameter schema compliance
        missing_params = check_gt_param_compliance(sample, tool_map)
        if missing_params:
            issues["gt_missing_required_param"].append({"id": tid, "missing_params": missing_params})

        # 4. Role sequence
        role_issues = check_role_sequence(sample)
        if role_issues:
            issues["bad_role_sequence"].append({"id": tid, "issues": role_issues})

        # 6. Duplicate trajectory_id
        if tid in seen_ids:
            issues["duplicates"].append(
                {"id": tid, "type": "duplicate_trajectory_id", "first_at_index": seen_ids[tid]}
            )
        else:
            seen_ids[tid] = idx

        # 6b. Duplicate content fingerprint
        fp = content_fingerprint(sample)
        if fp in seen_fps:
            issues["duplicates"].append(
                {"id": tid, "type": "duplicate_content", "matches_id": seen_fps[fp]}
            )
        else:
            seen_fps[fp] = tid

        # 7. Reward sanity
        reward = check_reward_sanity(sample)
        if reward is not None:
            reward_values.append(reward)
            if reward < 0:
                issues["gt_negative_reward"].append({"id": tid, "reward": round(reward, 4)})

        # 8. Distribution (required fields only; optional metadata tracked if present)
        n_gt  = len(sample.get("ground_truth_calls", []))
        n_msg = len(sample.get("prompt_messages", []))
        dist["by_gt_call_count"][str(n_gt)] = dist["by_gt_call_count"].get(str(n_gt), 0) + 1
        dist["by_message_count"][str(n_msg)] = dist["by_message_count"].get(str(n_msg), 0) + 1

        # optional metadata
        cat = sample.get("category")
        dom = sample.get("domain")
        if cat:
            dist["by_category"][cat] = dist["by_category"].get(cat, 0) + 1
        if dom:
            dist["by_domain"][dom] = dist["by_domain"].get(dom, 0) + 1

    # Reward stats
    reward_stats: dict = {}
    if reward_values:
        reward_stats = {
            "mean":       round(sum(reward_values) / len(reward_values), 4),
            "min":        round(min(reward_values), 4),
            "max":        round(max(reward_values), 4),
            "below_zero": sum(1 for r in reward_values if r < 0),
            "below_three": sum(1 for r in reward_values if r < 3.0),
        }

    # Deduplicate issue lists by id to avoid double-counting
    issue_counts = {k: len(v) for k, v in issues.items()}
    total_flags = sum(issue_counts.values())

    # Affected sample ids (unique)
    affected_ids: set[str] = set()
    for v in issues.values():
        for entry in v:
            if isinstance(entry, dict):
                affected_ids.add(entry.get("id", ""))
            # parse errors are strings, not dicts

    pass_rate = round(1 - len(affected_ids) / max(len(samples), 1), 4)

    summary = {
        "total_samples":      len(samples),
        "affected_samples":   len(affected_ids),
        "total_issue_flags":  total_flags,
        "pass_rate":          pass_rate,
        "checks":             issue_counts,
        "reward_stats":       reward_stats,
    }

    return {
        "data_path":   str(data_path),
        "summary":     summary,
        "issues":      issues,
        "distribution": dist,
    }


def print_report(report: dict):
    s = report["summary"]
    print(f"\n{'='*62}")
    print(f"AUDIT REPORT — {report['data_path']}")
    print(f"{'='*62}")
    print(f"  Total samples    : {s['total_samples']}")
    print(f"  Affected samples : {s['affected_samples']}")
    print(f"  Total flags      : {s['total_issue_flags']}")
    print(f"  Pass rate        : {s['pass_rate']*100:.1f}%")
    print()

    print("  Issue counts by check:")
    for check, count in s["checks"].items():
        bar = "  ✓" if count == 0 else f"  ✗ {count}"
        print(f"    {check:<38} {bar}")

    if s["reward_stats"]:
        rs = s["reward_stats"]
        print()
        print("  Reward sanity (GT mock completions):")
        print(f"    mean={rs['mean']}, min={rs['min']}, max={rs['max']}")
        print(f"    below 0.0: {rs['below_zero']},  below 3.0: {rs['below_three']}")

    print()
    print("  Distribution:")
    for key, val in report["distribution"].items():
        if isinstance(val, dict):
            sorted_items = sorted(val.items(), key=lambda x: -x[1] if isinstance(x[1], int) else 0)
            print(f"    {key}:")
            for subk, cnt in sorted_items:
                print(f"      {subk}: {cnt}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data quality audit for tool-learning JSONL datasets.")
    parser.add_argument("--data", default="data/dataset_14_may.jsonl",
                        help="Path to JSONL dataset (default: data/dataset_14_may.jsonl)")
    parser.add_argument("--out", default=None,
                        help="Path to write JSON report (default: data/quality_report.json)")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: data file not found: {data_path}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else data_path.parent / "quality_report.json"

    report = run_audit(str(data_path))
    print_report(report)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report written to {out_path}")


if __name__ == "__main__":
    main()
