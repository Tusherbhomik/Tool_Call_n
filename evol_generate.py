#!/usr/bin/env python3
"""
evol_generate.py — Two-step WizardLM Evol-Instruct for tool-learning datasets.

Faithful to the WizardLM paper (Xu et al., 2023):

  STEP 1 — Evolve the user's NATURAL LANGUAGE request (pure text, no JSON,
            no schema pressure). The LLM rewrites freely, just like WizardLM
            evolves plain-text instructions. Six operations mirror the paper:
            five in-depth (AddConstraint, Deepen, Concretize, Parallel, Context)
            and one in-breadth (NewRequest).

  STEP 2 — Given the evolved conversation + tools, generate ground_truth_calls
            FRESH by reasoning (not mutating the old GT mechanically). The LLM
            sees the full context and decides which tool to call and with what
            arguments — exactly how BFCL evaluates models.

Output: 4-field JSONL  {trajectory_id, prompt_messages, tools, ground_truth_calls}

Workflow (iterate until you reach --target):
    python evol_generate.py --batch 20 --target 20 --api-key <key>
    # review 20 samples manually
    python evol_generate.py --batch 100 --target 5000 --api-key <key>
    # auto-audit per batch, pick up where it left off
"""

import json, argparse, sys, random, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("ERROR: openai package not installed.  Run: pip install openai"); sys.exit(1)

try:
    from reward import toolrl_reward
    _REWARD_AVAILABLE = True
except Exception as e:
    print(f"[WARN] reward.py unavailable — reward sanity filter skipped."); _REWARD_AVAILABLE = False

try:
    from audit_dataset import run_audit
    _AUDIT_AVAILABLE = True
except Exception as e:
    print(f"[WARN] audit_dataset.py unavailable — inline audit skipped."); _AUDIT_AVAILABLE = False

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL   = "google/gemini-flash-1.5"
KEEP_FIELDS     = {"trajectory_id", "prompt_messages", "tools", "ground_truth_calls"}

ALL_OPS = ["ADD_CONSTRAINT", "DEEPEN", "CONCRETIZE", "PARALLEL_CALLS",
           "COMPLICATE_CONTEXT", "NEW_REQUEST"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _last_user_msg(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "").strip()
    return ""


def _tool_brief(tools: list[dict]) -> str:
    """One line per tool: name + description + required params."""
    lines = []
    for t in tools:
        fn   = t.get("function", {})
        name = fn.get("name", "")
        desc = fn.get("description", "").split(".")[0]
        req  = fn.get("parameters", {}).get("required", [])
        suffix = f"  [needs: {', '.join(req)}]" if req else ""
        lines.append(f"  • {name}: {desc}{suffix}")
    return "\n".join(lines)


def _tool_map(tools: list[dict]) -> dict[str, dict]:
    return {t["function"]["name"]: t["function"].get("parameters", {})
            for t in tools if "function" in t and "name" in t["function"]}


def _format_conv(messages: list[dict]) -> str:
    """Human-readable conversation for Step 2 prompt."""
    out = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            out.append(f"[SYSTEM]: {content}")
        elif role == "user":
            out.append(f"[USER]: {content}")
        elif role == "assistant":
            tcs = m.get("tool_calls", [])
            if tcs:
                for tc in tcs:
                    fn   = tc.get("function", {})
                    name = fn.get("name", "")
                    try:   args = json.loads(fn.get("arguments", "{}"))
                    except: args = {}
                    out.append(f"[ASSISTANT → {name}]: {json.dumps(args)}")
            elif content:
                out.append(f"[ASSISTANT]: {content}")
        elif role == "tool":
            out.append(f"[TOOL RESULT]: {content}")
    return "\n\n".join(out)


def _extract_json_array(text: str) -> list | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            return obj if isinstance(obj, list) else None
        except json.JSONDecodeError:
            pass
    return None


def _extract_json_obj(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1: return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:   esc = False; continue
        if ch == "\\": esc = True; continue
        if ch == '"':  in_str = not in_str; continue
        if in_str: continue
        if ch == "{":  depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i+1])
                    return obj if isinstance(obj, dict) else None
                except: return None
    return None


def _call(client, model, system, user, max_retries=3) -> str | None:
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=1024,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": user}],
            )
            return r.choices[0].message.content or ""
        except RateLimitError:
            time.sleep(5 * 2**attempt)
        except APIError as e:
            if attempt == max_retries - 1:
                print(f"  [API] {e}", flush=True)
            time.sleep(2)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Evolve the natural language request
# ─────────────────────────────────────────────────────────────────────────────

_S1_SYSTEM = """\
You are a dataset curator improving training data for AI assistants.
Rewrite the given user request using the specified method.
Write naturally — as a real human would phrase it.
Do NOT mention tool names, API names, or technical details in your output.
Output ONLY the rewritten request text. Nothing else."""


_S1_OPS = {

"ADD_CONSTRAINT": """\
METHOD — Add Constraint (WizardLM in-depth #1)
Add EXACTLY ONE new realistic constraint or requirement. Good types:
  • Deadline    : "…by 3 pm" / "…before the standup"
  • Exclusion   : "…but avoid Mondays" / "…don't overlap with lunch"
  • Format pref : "…with a 15-min buffer before and after"
  • Notification: "…and remind me 30 minutes early"

Available tools in this domain (do NOT name these in output):
{tool_brief}

Original request:
"{last_user}"

Rewritten request with exactly one new constraint (natural, one sentence):""",

"DEEPEN": """\
METHOD — Deepen (WizardLM in-depth #2)
Make the request more specific and precise. Replace vague values with exact ones:
  • "tomorrow"       → a real date/time  e.g. "next Tuesday at 2 pm"
  • "the team"       → specific names    e.g. "Alice, Bob, and Carol"
  • "a quick chat"   → "a 20-minute sync"
  • "the usual"      → an exact figure   e.g. "$1,500"

Available tools in this domain (do NOT name these in output):
{tool_brief}

Original request:
"{last_user}"

More specific, precise version (natural, one sentence):""",

"CONCRETIZE": """\
METHOD — Concretize (WizardLM in-depth #3)
Replace abstract / placeholder entities with specific realistic ones:
  • "John"         → "Marcus Webb (mwebb@company.io)"
  • "the product"  → "Horizon Analytics v2.4"
  • "our server"   → "prod-postgres-01.us-east"
  • "tomorrow"     → a near-future date like "May 20th"
  • Generic event  → "Q2 OKR retrospective"

Available tools in this domain (do NOT name these in output):
{tool_brief}

Original request:
"{last_user}"

More concrete version with real specifics (natural, one or two sentences):""",

"PARALLEL_CALLS": """\
METHOD — Parallel Calls (WizardLM in-depth #4 — multi-step reasoning)
Rewrite the request so it clearly asks for TWO things at the same time.
The two actions should be independent (not one depending on the other).
Examples:
  • "Book a flight to Paris and reserve a hotel for the same dates"
  • "Schedule a meeting with Alice and send her the agenda document"
  • "Set a price alert for AAPL at $180 and buy 5 shares of TSLA now"

Available tools in this domain (do NOT name these in output):
{tool_brief}

Original request:
"{last_user}"

Rewritten to require two parallel actions (natural, one sentence):""",

"COMPLICATE_CONTEXT": """\
METHOD — Complicate Input (WizardLM in-depth #5)
You will create a short prior exchange that gives context, then a new final
request that depends on that context.

Output a JSON object with exactly these 3 keys:
{{
  "prior_user"      : "a realistic prior user message (1-2 sentences)",
  "prior_assistant" : "a natural assistant reply — plain text, no tool call (1-2 sentences)",
  "final_request"   : "the new final user message that references what was just said"
}}

The final_request must depend on or refer back to the prior exchange.
Do NOT mention tool names anywhere.

Available tools in this domain:
{tool_brief}

Original final request:
"{last_user}"

Output only the JSON object:""",

"NEW_REQUEST": """\
METHOD — New Request (WizardLM in-breadth)
Write a completely NEW user request for this same tool domain.
It should be a different use case — more rare or specific than the original.
No connection to the original request is needed.
Write as a real user would: natural, direct, one or two sentences.
Do NOT mention tool names or technical details.

Tools available in this domain:
{tool_brief}

Original request (for domain reference only):
"{last_user}"

A brand-new, distinct request a real user might make with these same tools:""",
}


def step1_evolve(client, model, seed: dict, operation: str) -> str | dict | None:
    """
    Returns:
      - str  for text operations (the evolved user message)
      - dict for COMPLICATE_CONTEXT  {"prior_user", "prior_assistant", "final_request"}
      - None on failure
    """
    last_user  = _last_user_msg(seed["prompt_messages"])
    tool_brief = _tool_brief(seed["tools"])
    tmpl       = _S1_OPS[operation]
    user_msg   = tmpl.format(last_user=last_user, tool_brief=tool_brief)

    raw = _call(client, model, _S1_SYSTEM, user_msg)
    if not raw:
        return None

    raw = raw.strip()

    if operation == "COMPLICATE_CONTEXT":
        obj = _extract_json_obj(raw)
        if not obj:
            return None
        if not all(k in obj for k in ("prior_user", "prior_assistant", "final_request")):
            return None
        return obj

    # Plain text — basic sanity: not empty, not too long, not same as original
    if not raw or len(raw) < 5:
        return None
    if raw.lower().strip('"') == last_user.lower():
        return None   # no change
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Build new prompt_messages from step-1 result
# ─────────────────────────────────────────────────────────────────────────────

def build_messages(seed_messages: list[dict], step1_result, operation: str) -> list[dict] | None:
    """
    Returns new prompt_messages list ending with a user message.
    """
    if operation == "NEW_REQUEST":
        # In-breadth: fresh single-turn — keep only system message
        system = next((m for m in seed_messages if m.get("role") == "system"), None)
        if not system:
            return None
        return [system, {"role": "user", "content": str(step1_result)}]

    if operation == "COMPLICATE_CONTEXT":
        # Insert prior exchange before the last user message, replace it with final_request
        prior_user      = step1_result.get("prior_user", "").strip()
        prior_assistant = step1_result.get("prior_assistant", "").strip()
        final_request   = step1_result.get("final_request", "").strip()
        if not prior_user or not prior_assistant or not final_request:
            return None

        # Find last user message index
        last_user_idx = None
        for i in range(len(seed_messages) - 1, -1, -1):
            if seed_messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return None

        new_msgs = (seed_messages[:last_user_idx]
                    + [{"role": "user",      "content": prior_user},
                       {"role": "assistant", "content": prior_assistant},
                       {"role": "user",      "content": final_request}])
        return new_msgs

    # All other operations: replace last user message
    last_user_idx = None
    for i in range(len(seed_messages) - 1, -1, -1):
        if seed_messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        return None

    new_msgs = list(seed_messages)
    new_msgs[last_user_idx] = {"role": "user", "content": str(step1_result)}
    return new_msgs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Generate ground_truth_calls fresh
# ─────────────────────────────────────────────────────────────────────────────

_S2_SYSTEM = """\
You are a precise function-calling assistant.
Given a conversation and the available tools, decide what tool call(s) to make next.

Rules:
• Output ONLY a valid JSON array — no explanation, no markdown, no code fences.
• Format: [{"name": "tool_name", "arguments": {"param": value, ...}}]
• Use EXACT tool names from the list — do not invent tools.
• Include ALL required parameters for each call.
• If the request needs no tool (clarification or direct answer): output []
• If two things are asked in parallel, include both calls in the array."""


def step2_generate_gt(client, model, tools: list[dict],
                      messages: list[dict]) -> list[dict] | None:
    tools_json = json.dumps(tools, indent=2)
    conv_text  = _format_conv(messages)

    user_msg = (
        f"Available tools:\n{tools_json}\n\n"
        f"Conversation (respond to the final [USER] message):\n{conv_text}\n\n"
        f"Output the JSON array of tool call(s):"
    )

    raw = _call(client, model, _S2_SYSTEM, user_msg, max_retries=3)
    if not raw:
        return None

    arr = _extract_json_array(raw)
    if arr is None:
        return None

    # Normalize: accept both "arguments" and "parameters"
    result = []
    for item in arr:
        if not isinstance(item, dict) or "name" not in item:
            continue
        args = item.get("arguments") or item.get("parameters") or {}
        if not isinstance(args, dict):
            args = {}
        result.append({"name": item["name"], "arguments": args})

    return result   # [] is valid (no-tool scenario)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate(sample: dict, seed: dict) -> list[str]:
    reasons = []
    for f in KEEP_FIELDS:
        if f not in sample:
            reasons.append(f"missing field: {f}")
    if reasons:
        return reasons

    tmap = _tool_map(sample["tools"])
    gt   = sample["ground_truth_calls"]

    for call in gt:
        name = call.get("name", "")
        if name and name not in tmap:
            reasons.append(f"GT tool '{name}' not in tools")

    for call in gt:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not isinstance(args, dict): args = {}
        for param in tmap.get(name, {}).get("required", []):
            if param not in args:
                reasons.append(f"GT '{name}' missing required param '{param}'")

    # Evolved must differ from seed (at minimum the last user message changed)
    evolved_last = _last_user_msg(sample["prompt_messages"])
    seed_last    = _last_user_msg(seed["prompt_messages"])
    if evolved_last == seed_last:
        reasons.append("last user message unchanged from seed")

    if _REWARD_AVAILABLE and not reasons:
        norm_gt = [{"name": c["name"], "arguments": c.get("arguments", {})} for c in gt]
        if norm_gt:
            calls_str  = "".join(f'<tool_call>{json.dumps(c)}</tool_call>' for c in norm_gt)
            completion = f"<think>Calling tools.</think>{calls_str}"
        else:
            completion = "<think>No tool needed.</think><response>Done.</response>"
        if toolrl_reward(completion, norm_gt) < 0:
            reasons.append("GT failed reward sanity")

    return reasons


def _fingerprint(sample: dict) -> str:
    last = _last_user_msg(sample["prompt_messages"])
    names = ",".join(sorted(c.get("name", "") for c in sample.get("ground_truth_calls", [])))
    return f"{last}||{names}"


# ─────────────────────────────────────────────────────────────────────────────
# Worker (one task = one evolved sample)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    client, model, seed, operation = args

    # Step 1 — evolve natural language request
    s1 = step1_evolve(client, model, seed, operation)
    if s1 is None:
        return None, operation, "step1_fail"

    # Build new prompt_messages
    new_msgs = build_messages(seed["prompt_messages"], s1, operation)
    if new_msgs is None:
        return None, operation, "msg_build_fail"

    # Step 2 — generate GT calls fresh
    gt = step2_generate_gt(client, model, seed["tools"], new_msgs)
    if gt is None:
        return None, operation, "step2_fail"

    # Assemble 4-field sample
    sample = {
        "trajectory_id":   f"{seed['trajectory_id']}_evol_{operation}",
        "prompt_messages": new_msgs,
        "tools":           seed["tools"],
        "ground_truth_calls": gt,
    }

    reasons = _validate(sample, seed)
    if reasons:
        return None, operation, reasons[0]

    return sample, operation, "ok"


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists(): return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except json.JSONDecodeError: pass
    return out


def _count(path: Path) -> int:
    if not path.exists(): return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip(): n += 1
    return n


def _load_fps(path: Path) -> set[str]:
    return {_fingerprint(s) for s in _load_jsonl(path)}


# ─────────────────────────────────────────────────────────────────────────────
# Inline audit
# ─────────────────────────────────────────────────────────────────────────────

def _inline_audit(batch_path: Path):
    if not _AUDIT_AVAILABLE or not batch_path.exists(): return
    print(f"\n── Audit: {batch_path.name} ──────────────────────────")
    report = run_audit(str(batch_path))
    s = report["summary"]
    print(f"  Samples : {s['total_samples']}  |  Pass rate : {s['pass_rate']*100:.1f}%")
    for check, count in s["checks"].items():
        if count:
            print(f"  ✗ {check}: {count}")
    if s.get("reward_stats"):
        rs = s["reward_stats"]
        print(f"  Reward on GT: mean={rs['mean']}  min={rs['min']}")
    print("─" * 54)


def _print_op_stats(op_stats: dict):
    print("\n  Per-operation results:")
    for op in ALL_OPS:
        st  = op_stats.get(op, {})
        ok  = st.get("ok", 0)
        fail= st.get("fail", 0)
        tot = ok + fail
        pct = f"{100*ok//tot}%" if tot else "—"
        print(f"    {op:<22} ok={ok:3d}  fail={fail:3d}  pass={pct}")


# ─────────────────────────────────────────────────────────────────────────────
# Main generation loop
# ─────────────────────────────────────────────────────────────────────────────

def generate(input_path, output_path, target, batch_size, model,
             max_workers, operations, auto, api_key, base_url, seed_limit):

    import os
    client = OpenAI(api_key=api_key or os.environ.get("OPENROUTER_API_KEY"),
                    base_url=base_url)

    seeds = _load_jsonl(Path(input_path))
    if seed_limit:
        seeds = seeds[:seed_limit]
    print(f"Seeds : {len(seeds)}  |  Operations : {operations}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_fps = _load_fps(out_path)
    for s in seeds:
        existing_fps.add(_fingerprint(s))   # never duplicate a seed

    out_lock  = Lock()
    batch_num = 0

    while True:
        current = _count(out_path)
        remaining = target - current
        if remaining <= 0:
            print(f"\nTarget {target} reached."); break

        this_batch = min(batch_size, remaining)
        batch_num += 1

        print(f"\n{'='*58}")
        print(f"Batch {batch_num}  |  generating {this_batch}  |  model: {model}")
        print(f"Output so far: {current}/{target}")
        print(f"{'='*58}")

        batch_tmp = out_path.parent / f"_batch_{batch_num:04d}_tmp.jsonl"
        op_stats: dict[str, dict] = {op: {"ok": 0, "fail": 0} for op in operations}
        n_written = 0

        tasks = [(client, model, s, op) for s in seeds for op in operations]
        random.shuffle(tasks)

        class DualWriter:
            def __init__(self, master, tmp):
                self.master, self.tmp = master, tmp
            def write(self, s):
                self.master.write(s); self.tmp.write(s)
            def flush(self):
                self.master.flush(); self.tmp.flush()

        with open(out_path, "a") as mf, open(batch_tmp, "w") as tf:
            dw = DualWriter(mf, tf)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_worker, t): t for t in tasks}
                for fut in as_completed(futures):
                    if n_written >= this_batch:
                        for f in futures: f.cancel()
                        break

                    evolved, op, reason = fut.result()
                    if evolved is None:
                        op_stats[op]["fail"] += 1
                        continue

                    fp = _fingerprint(evolved)
                    with out_lock:
                        if fp in existing_fps:
                            op_stats[op]["fail"] += 1
                            continue
                        existing_fps.add(fp)
                        op_stats[op]["ok"] += 1
                        n_written += 1

                    with out_lock:
                        dw.write(json.dumps(evolved) + "\n")
                        dw.flush()

                    if n_written % 10 == 0 or n_written == this_batch:
                        print(f"  [{n_written}/{this_batch}]", flush=True)

        print(f"\nBatch {batch_num} done — wrote {n_written} new samples.")
        _print_op_stats(op_stats)

        if n_written > 0:
            _inline_audit(batch_tmp)
        batch_tmp.unlink(missing_ok=True)

        total_now = _count(out_path)
        print(f"\nOutput file: {out_path}  ({total_now}/{target} samples)")

        if not auto:
            print("\n[Stopped. Review the audit above, fix if needed, then re-run.]")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Two-step WizardLM Evol-Instruct for tool-learning datasets.")
    ap.add_argument("--input",    default="data/dataset_clean_4field.jsonl")
    ap.add_argument("--output",   default="data/dataset_expanded.jsonl")
    ap.add_argument("--target",   type=int, default=20,
                    help="Total samples desired in output file (default: 20)")
    ap.add_argument("--batch",    type=int, default=20,
                    help="Samples per run before stopping for review (default: 20)")
    ap.add_argument("--auto",     action="store_true",
                    help="Keep running batches until --target (no stop between batches)")
    ap.add_argument("--model",    default=DEFAULT_MODEL)
    ap.add_argument("--api-key",  default=None)
    ap.add_argument("--base-url", default=OPENROUTER_BASE)
    ap.add_argument("--workers",  type=int, default=10,
                    help="Max concurrent API calls (default: 10; each sample = 2 calls)")
    ap.add_argument("--ops",      default=None,
                    help=f"Comma-separated ops (default: all). Choices: {ALL_OPS}")
    ap.add_argument("--seed-limit", type=int, default=None,
                    help="Use only first N seeds (useful for testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Clear output file before starting")
    ap.add_argument("--rand-seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.rand_seed)

    ops = ALL_OPS[:]
    if args.ops:
        ops = [o.strip().upper() for o in args.ops.split(",")]
        bad = [o for o in ops if o not in ALL_OPS]
        if bad:
            print(f"ERROR: unknown ops {bad}. Valid: {ALL_OPS}"); sys.exit(1)

    if not Path(args.input).exists():
        print(f"ERROR: input not found: {args.input}"); sys.exit(1)

    if args.overwrite and Path(args.output).exists():
        Path(args.output).unlink()
        print("Output file cleared.")

    import os
    if not (args.api_key or os.environ.get("OPENROUTER_API_KEY")):
        print("ERROR: provide --api-key or set OPENROUTER_API_KEY env var."); sys.exit(1)

    generate(
        input_path=args.input,
        output_path=args.output,
        target=args.target,
        batch_size=args.batch,
        model=args.model,
        max_workers=args.workers,
        operations=ops,
        auto=args.auto,
        api_key=args.api_key,
        base_url=args.base_url,
        seed_limit=args.seed_limit,
    )


if __name__ == "__main__":
    main()
