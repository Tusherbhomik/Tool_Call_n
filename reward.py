"""
reward.py — ToolRL Reward Design (paper-faithful implementation)

Implements exactly the reward described in:
  "ToolRL: Reward is All Tool Learning Needs" (Qian et al., 2025)

Two components only:
  R_final = R_format + R_correct  ∈ [-3, 4]

──────────────────────────────────────────────────────────────────────────────
Level 0 — FORMAT REWARD  R_format ∈ {0, 1}  (Section 3.3)
  Checks whether model output contains all required special tokens
  (<think>, <tool_call>, <response>) in the correct order as specified
  by the ground truth.

Level 1 — CORRECTNESS REWARD  R_correct ∈ [-3, 3]  (Section 3.3)
  Evaluates predicted tool calls P against ground-truth calls G via
  three sub-components:

    r_name  = |N_G ∩ N_P| / |N_G ∪ N_P|          ∈ [0, 1]
    r_param = Σ_j |keys(G_j) ∩ keys(P_j)| /
                   |keys(G_j) ∪ keys(P_j)|         ∈ [0, |G|]
    r_value = Σ_j Σ_{k ∈ keys(G_j)} 1[G_j[k] = P_j[k]]
                                                   ∈ [0, Σ_j |keys(G_j)|]

  Optimal matching between P and G via Hungarian algorithm to maximise
  the total match score:

    S_max    = 1 + |G| + Σ_j |keys(G_j)|
    R_max    = r_name + best-matched (r_param + r_value)
    R_correct = 6 * R_max / S_max - 3

No quality bonuses, no sigmoid gate, no irrelevance/efficiency/dependency
terms — those are not in the paper.
"""

import re
import json
from scipy.optimize import linear_sum_assignment


# ══════════════════════════════════════════════════════════════════════════════
# PARSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_tool_calls(completion: str) -> list[dict]:
    """
    Extract tool calls from a model completion string.

    Supports two formats used in the paper:
      1. <tool_call>{"name": ..., "parameters": {...}}</tool_call>
      2. Bare JSON objects at top level (fallback)

    Returns a list of dicts, each with at least "name" and either
    "arguments" or "parameters" key.
    """
    calls = []

    # Primary: look for content inside <tool_call>...</tool_call> tags.
    tag_pattern = re.compile(
        r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
    )
    blocks = tag_pattern.findall(completion)

    if blocks:
        for block in blocks:
            block = block.strip()
            # Each block may contain multiple JSON objects (one per line).
            for line in block.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "name" in obj:
                        calls.append(obj)
                except json.JSONDecodeError:
                    pass
        if calls:
            return calls

    # Fallback: try to parse the whole completion as a single JSON object.
    try:
        obj = json.loads(completion.strip())
        if isinstance(obj, dict) and "name" in obj:
            return [obj]
    except json.JSONDecodeError:
        pass

    return calls


def _get_args(call: dict) -> dict:
    """
    Return the arguments dict from a call, regardless of key name.
    Guards against malformed model output where arguments is a string
    or other non-dict type instead of a JSON object.
    """
    args = call.get("arguments") or call.get("parameters") or {}
    if not isinstance(args, dict):
        return {}
    return args


def _norm_value(v) -> str:
    """Normalise a parameter value to a lowercase stripped string."""
    if v is None:
        return ""
    return str(v).strip().lower()


# ══════════════════════════════════════════════════════════════════════════════
# FORMAT REWARD  (Section 3.3, Format Reward)
# ══════════════════════════════════════════════════════════════════════════════

def _tool_call_json_valid(completion: str) -> bool:
    """
    Return True if at least one <tool_call> block contains a strictly valid JSON
    object with a "name" key.  No brace-stripping tolerance — the reward must
    penalise malformed output so the model is forced to learn correct JSON.
    (Brace-stripping leniency lives only in the evaluation handler.)
    """
    match = re.search(r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL | re.IGNORECASE)
    if not match:
        return False
    for line in match.group(1).strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "name" in obj:
                return True
        except json.JSONDecodeError:
            continue
    return False


def format_reward(completion: str, gt: list[dict]) -> float:
    """
    R_format ∈ {0, 1}

    Matched to the actual training data format (OpenAI tool_calls style rendered
    through Qwen's chat template — no <think> or <response> tags):

      gt non-empty  →  R_format = 1 if <tool_call>...</tool_call> is present
                                     AND the JSON inside is parseable with a
                                     "name" key; 0 otherwise.
      gt empty      →  R_format = 1 if no <tool_call> tag is produced (model
                                     answers directly); 0 if a spurious tool
                                     call appears.
    """
    has_tool_call = bool(re.search(r"<tool_call>.*?</tool_call>", completion, re.DOTALL | re.IGNORECASE))

    if gt:
        # Tool call expected — tag must exist and contain valid JSON.
        if not has_tool_call:
            return 0.0
        if not _tool_call_json_valid(completion):
            return 0.0
    else:
        # No tool call expected — penalise spurious tool calls.
        if has_tool_call:
            return 0.0

    return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# CORRECTNESS REWARD  (Section 3.3, Correctness Reward)
# ══════════════════════════════════════════════════════════════════════════════

def correctness_reward(completion: str, gt: list[dict]) -> float:
    """
    R_correct ∈ [-3, 3]

    Sub-components (paper equations):
      r_name  = |N_G ∩ N_P| / |N_G ∪ N_P|
      r_param = Σ_j |keys(G_j) ∩ keys(P_j)| / |keys(G_j) ∪ keys(P_j)|
      r_value = Σ_j Σ_{k ∈ keys(G_j)} 1[G_j[k] = P_j[k]]

    Optimal matching via Hungarian algorithm on (r_param + r_value).
    S_max = 1 + |G| + Σ_j |keys(G_j)|
    R_correct = 6 * R_max / S_max - 3
    """
    pred = _parse_tool_calls(completion)

    # Edge cases: no ground-truth calls expected.
    if not gt:
        return 3.0 if not pred else -3.0
    if not pred:
        return -3.0

    n_gt   = len(gt)
    n_pred = len(pred)

    # ── Tool Name Matching (set-level Jaccard) ────────────────────────────────
    gt_names   = {g.get("name", "").lower() for g in gt   if g.get("name")}
    pred_names = {p.get("name", "").lower() for p in pred if p.get("name")}
    union_names = gt_names | pred_names
    r_name = len(gt_names & pred_names) / len(union_names) if union_names else 1.0

    # ── S_max ─────────────────────────────────────────────────────────────────
    total_gt_params = sum(len(_get_args(g)) for g in gt)
    s_max = 1.0 + n_gt + total_gt_params
    if s_max == 0:
        return 3.0

    # ── Hungarian Matching on (r_param + r_value) per pair ───────────────────
    size = max(n_gt, n_pred)
    cost = [[0.0] * size for _ in range(size)]

    for i in range(n_gt):
        gt_args  = _get_args(gt[i])
        gt_keys  = set(gt_args.keys())

        for j in range(n_pred):
            pred_args = _get_args(pred[j])
            pred_keys = set(pred_args.keys())

            # r_param contribution for this pair
            union_keys = gt_keys | pred_keys
            pair_param = (
                len(gt_keys & pred_keys) / len(union_keys)
                if union_keys else 1.0
            )

            # r_value contribution for this pair
            pair_value = sum(
                1.0
                for k in gt_keys
                if k in pred_args
                and _norm_value(pred_args[k]) == _norm_value(gt_args[k])
            )

            cost[i][j] = -(pair_param + pair_value)  # minimise negative

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_param_value = sum(
        -cost[i][j]
        for i, j in zip(row_ind, col_ind)
        if i < n_gt and j < n_pred
    )

    r_max     = r_name + matched_param_value
    r_correct = 6.0 * (r_max / s_max) - 3.0
    return max(-3.0, min(3.0, r_correct))


# ══════════════════════════════════════════════════════════════════════════════
# FINAL REWARD  (Section 3.3)
# ══════════════════════════════════════════════════════════════════════════════

def toolrl_reward(completion: str, gt: list[dict]) -> float:
    """
    R_final = R_format + R_correct  ∈ [-3, 4]

    Args:
        completion: Raw model output string (may contain <think>, <tool_call>,
                    <response> tags).
        gt:         List of ground-truth tool call dicts, each with
                    "name" and "arguments"/"parameters" keys.
                    Pass an empty list when no tool call is expected.

    Returns:
        Scalar reward in [-3, 4].
    """
    r_fmt = format_reward(completion, gt)

    # Paper design: correctness is computed regardless of format gate.
    # (The paper does not describe a hard gate; both components always sum.)
    r_cor = correctness_reward(completion, gt)

    return r_fmt + r_cor


# ══════════════════════════════════════════════════════════════════════════════
# GRPO ADAPTER — matches your training sample schema
# ══════════════════════════════════════════════════════════════════════════════

def reward_from_sample(sample: dict, completion: str) -> float:
    """
    Convenience wrapper for your JSONL training samples.

    Expected sample keys (matching the schema you showed):
      "ground_truth_calls": list of {name, arguments} dicts

    Args:
        sample:     One parsed training sample dict.
        completion: The model's raw output string for that sample.

    Returns:
        Scalar reward in [-3, 4].
    """
    gt = sample.get("ground_truth_calls", [])
    # Normalise key: some samples use "arguments", some "parameters"
    normalised_gt = []
    for call in gt:
        normalised_gt.append({
            "name": call.get("name", ""),
            "arguments": call.get("arguments") or call.get("parameters") or {},
        })
    return toolrl_reward(completion, normalised_gt)


# ══════════════════════════════════════════════════════════════════════════════
# SANITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def _run_sanity_tests():
    """
    Verify the reward function against expected values derived from the
    paper's formulas.

    Perfect single call:
      r_name  = 1/1 = 1.0
      r_param = 3/3 = 1.0   (3 matching keys)
      r_value = 3            (3 exact value matches)
      R_max   = 1 + 1 + 3 = 5
      S_max   = 1 + 1 + 3 = 5
      R_correct = 6*(5/5) - 3 = 3.0
      R_format  = 1.0
      R_final   = 4.0

    Wrong tool (no name match, no param match):
      r_name  = 0/2 = 0.0
      r_param = 0/3 = 0.0
      r_value = 0
      R_max   = 0
      R_correct = 6*(0/5) - 3 = -3.0
      R_format  = 1.0   (format is valid)
      R_final   = -2.0

    No tool when one expected:
      R_format  = 0.0  (missing <tool_call>)
      R_correct = -3.0 (pred is empty, gt is not)
      R_final   = -3.0

    Empty gt + no call (correct clarification):
      R_format  = 1.0
      R_correct = 3.0  (both empty)
      R_final   = 4.0

    Empty gt + spurious call:
      R_format  = 0.0  (tool_call present but not expected → no <response>)
      R_correct = -3.0 (pred non-empty, gt empty)
      R_final   = -3.0
    """
    gt_single = [{"name": "deploy_service",
                  "arguments": {"service_name": "payment-service",
                                "environment": "staging",
                                "version": "2.3.2"}}]

    # 1. Perfect single call
    # No <think> tags — model output is plain <tool_call> as trained
    perfect = (
        '<tool_call>{"name": "deploy_service", "arguments": '
        '{"service_name": "payment-service", "environment": "staging", "version": "2.3.2"}}'
        "</tool_call>"
    )
    r = toolrl_reward(perfect, gt_single)
    assert abs(r - 4.0) < 1e-9, f"perfect_single: expected 4.0 got {r}"

    # 2. Wrong tool (rollback_deployment instead of deploy_service)
    # r_name  = 0/2 = 0.0  (no name overlap)
    # Param matching: gt_keys={service_name,environment,version}
    #                 pred_keys={service_name,environment,target_version}
    #   intersection=2, union=4 → r_param=0.5
    #   r_value: service_name matches (payment-service=payment-service ✓),
    #            environment doesn't (prod≠staging), version key absent → 1
    # R_max = 0 + 0.5 + 1 = 1.5,  S_max = 5
    # R_correct = 6*(1.5/5) - 3 = -1.2
    # R_format  = 1.0  (format is valid)
    # R_final   = 1.0 + (-1.2) = -0.2
    wrong = (
        '<tool_call>{"name": "rollback_deployment", "arguments": '
        '{"service_name": "payment-service", "environment": "prod", "target_version": "2.3.1"}}'
        "</tool_call>"
    )
    r = toolrl_reward(wrong, gt_single)
    assert abs(r - (-0.2)) < 1e-9, f"wrong_tool: expected -0.2 got {r}"

    # 2b. Completely unrelated tool (no shared params at all)
    # r_name=0, r_param=0, r_value=0 → R_max=0, R_correct=-3, R_format=1 → total=-2
    unrelated = (
        '<tool_call>{"name": "trigger_ci_pipeline", "arguments": '
        '{"pipeline_name": "main", "branch": "dev"}}'
        "</tool_call>"
    )
    r = toolrl_reward(unrelated, gt_single)
    assert abs(r - (-2.0)) < 1e-9, f"unrelated_tool: expected -2.0 got {r}"

    # 3. No tool when one expected — plain text response, no <tool_call>
    no_tool = "I'm sorry, I can't help with that right now."
    r = toolrl_reward(no_tool, gt_single)
    assert abs(r - (-3.0)) < 1e-9, f"no_tool: expected -3.0 got {r}"

    # 4. Empty gt + correct clarification — plain text, no tool call
    clarify = "Could you provide more details so I can help you?"
    r = toolrl_reward(clarify, [])
    assert abs(r - 4.0) < 1e-9, f"empty_gt_correct: expected 4.0 got {r}"

    # 5. Empty gt + spurious call — model calls a tool when none expected
    spurious = '<tool_call>{"name": "deploy_service", "arguments": {}}</tool_call>'
    r = toolrl_reward(spurious, [])
    assert abs(r - (-3.0)) < 1e-9, f"empty_gt_spurious: expected -3.0 got {r}"

    # 6. Partial match — correct name, wrong version value
    partial = (
        '<tool_call>{"name": "deploy_service", "arguments": '
        '{"service_name": "payment-service", "environment": "staging", "version": "2.3.1"}}'
        "</tool_call>"
    )
    r = toolrl_reward(partial, gt_single)
    # r_name=1, r_param=3/3=1, r_value=2 (version wrong), R_max=4, S_max=5
    # R_correct = 6*(4/5)-3 = 1.8, R_format=1, total=2.8
    assert abs(r - 2.8) < 1e-9, f"partial_match: expected 2.8 got {r}"

    print("All sanity tests passed.")


if __name__ == "__main__":
    _run_sanity_tests()

    # ── Example: reward from your training sample schema ─────────────────────
    import json

    sample_json = """
    {
        "ground_truth_calls": [
            {"name": "deploy_service",
             "arguments": {"service_name": "payment-service",
                           "environment": "staging",
                           "version": "2.3.2"}}
        ]
    }
    """
    sample = json.loads(sample_json)

    model_output = (
        '<tool_call>{"name": "deploy_service", "arguments": '
        '{"service_name": "payment-service", "environment": "staging", "version": "2.3.2"}}'
        "</tool_call>"
    )

    reward = reward_from_sample(sample, model_output)
    print(f"Example reward: {reward}")   # Expected: 4.0