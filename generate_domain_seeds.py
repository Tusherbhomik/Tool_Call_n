#!/usr/bin/env python3
"""
generate_domain_seeds.py — Generate seed conversations for 8 BFCL-aligned domains
that are missing from the current dataset, and append them to the clean seed file.

Domains covered (matching BFCL v4 categories):
  math_science, file_system_os, maps_location, weather,
  food_dining, cloud_devops, web_search, database

Two-step per seed:
  Step A — generate tool schema JSON (once per domain, reused for all seeds)
  Step B — generate a realistic seed conversation using those tools

Each generated seed is validated with the same checks used in evol_generate.py.
Only passing seeds are appended to the output file.

Usage:
    python generate_domain_seeds.py --api-key <key>
    python generate_domain_seeds.py --api-key <key> --seeds-per-domain 30 --out data/dataset_clean_4field.jsonl
"""

import json
import argparse
import sys
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from openai import OpenAI, RateLimitError, APIError
except ImportError:
    print("ERROR: openai package not installed.  Run: pip install openai")
    sys.exit(1)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_MODEL   = "google/gemini-flash-1.5"
DEFAULT_OUT     = "data/dataset_clean_4field.jsonl"
SEEDS_PER_DOMAIN = 25


# ─────────────────────────────────────────────────────────────────────────────
# Domain definitions (BFCL-aligned)
# ─────────────────────────────────────────────────────────────────────────────

DOMAINS = {
    "math_science": {
        "bfcl_category": "Math & Science",
        "tools": [
            "calculate_expression", "convert_units", "solve_equation",
            "compute_statistics", "get_constant", "evaluate_integral",
        ],
        "n_tools": 5,
        "description": "mathematical computation, unit conversion, statistics, and scientific constants",
    },
    "file_system_os": {
        "bfcl_category": "File System & OS",
        "tools": [
            "read_file", "write_file", "list_directory", "move_file",
            "delete_file", "run_command", "get_file_info",
        ],
        "n_tools": 5,
        "description": "file reading/writing, directory listing, file operations, and shell commands",
    },
    "maps_location": {
        "bfcl_category": "Maps & Location",
        "tools": [
            "geocode_address", "search_nearby", "get_route", "get_distance",
            "get_timezone", "get_elevation",
        ],
        "n_tools": 5,
        "description": "geocoding, nearby search, routing, distance, and timezone lookup",
    },
    "weather": {
        "bfcl_category": "Weather & Environment",
        "tools": [
            "get_current_weather", "get_forecast", "get_hourly_forecast",
            "get_weather_alert", "get_air_quality",
        ],
        "n_tools": 4,
        "description": "current weather conditions, forecasts, hourly data, alerts, and air quality",
    },
    "food_dining": {
        "bfcl_category": "Food & Dining",
        "tools": [
            "search_restaurants", "get_menu", "place_order", "get_reviews",
            "make_reservation", "get_restaurant_details",
        ],
        "n_tools": 5,
        "description": "restaurant search, menus, food ordering, reviews, and reservations",
    },
    "cloud_devops": {
        "bfcl_category": "Cloud & DevOps",
        "tools": [
            "list_instances", "start_instance", "stop_instance", "get_logs",
            "create_bucket", "set_env_var", "deploy_service",
        ],
        "n_tools": 5,
        "description": "cloud instance management, log retrieval, storage buckets, environment variables, and deployments",
    },
    "web_search": {
        "bfcl_category": "Web & Security",
        "tools": [
            "web_search", "get_page_content", "extract_links",
            "check_url_safety", "summarize_page",
        ],
        "n_tools": 4,
        "description": "web search, page content retrieval, link extraction, and URL safety checking",
    },
    "database": {
        "bfcl_category": "Database & Storage",
        "tools": [
            "run_query", "insert_record", "update_record", "delete_record",
            "list_tables", "get_schema",
        ],
        "n_tools": 5,
        "description": "SQL queries, CRUD operations, table listing, and schema inspection",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# API helper
# ─────────────────────────────────────────────────────────────────────────────

def _call(client, model, system, user, max_retries=3) -> str | None:
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=2048,
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
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:    esc = False; continue
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
                except Exception:
                    return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step A — generate tool schema for a domain
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_GEN_SYSTEM = """\
You are generating tool definitions for a Berkeley Function Calling Leaderboard (BFCL)
evaluation dataset. Produce high-quality, unambiguous tool schemas.

Output ONLY a valid JSON array. No explanation, no markdown, no code fences."""


def generate_tool_schema(client, model, domain_key: str, domain_info: dict) -> list[dict] | None:
    n = domain_info["n_tools"]
    desc = domain_info["description"]
    suggested = ", ".join(domain_info["tools"])

    user_msg = f"""\
Generate a JSON array of {n} tool definitions for a {domain_key} AI assistant
covering {desc}.

Suggested function names (use most of them): {suggested}

Each tool must follow this schema exactly:
{{
  "type": "function",
  "function": {{
    "name": "<snake_case_name>",
    "description": "<clear, unambiguous description>",
    "parameters": {{
      "type": "object",
      "properties": {{
        "<param_name>": {{
          "type": "<string|integer|number|boolean|array>",
          "description": "<what this parameter means>"
        }}
      }},
      "required": ["<param1>", "<param2>"]
    }}
  }}
}}

Rules:
- Snake_case function names
- Realistic, specific parameter types
- 2–4 required parameters per tool
- Descriptions must be unambiguous (BFCL-quality)

Output only the JSON array of {n} tool objects."""

    raw = _call(client, model, _TOOL_GEN_SYSTEM, user_msg)
    if not raw:
        return None
    tools = _extract_json_array(raw)
    if not tools or len(tools) < 2:
        return None
    # Validate minimal structure
    valid = []
    for t in tools:
        if (isinstance(t, dict)
                and t.get("type") == "function"
                and "function" in t
                and "name" in t["function"]):
            valid.append(t)
    return valid if valid else None


# ─────────────────────────────────────────────────────────────────────────────
# Step B — generate one seed conversation
# ─────────────────────────────────────────────────────────────────────────────

_SEED_GEN_SYSTEM = """\
You are generating a realistic training sample for a function-calling AI assistant.

Output ONLY a valid JSON object. No explanation, no markdown, no code fences."""


def generate_seed_conversation(client, model, domain_key: str, tools: list[dict],
                               seed_index: int) -> dict | None:
    tools_json = json.dumps(tools, indent=2)
    tid = f"seed_{domain_key}_{seed_index:04d}"

    user_msg = f"""\
Generate a SHORT, realistic conversation between a user and a {domain_key} assistant.

The conversation should:
- Start with a system message describing the assistant
- Have 1–3 turns of natural back-and-forth (optional)
- End with a user message the assistant must respond to using one of the provided tools

Output a JSON object with exactly 4 fields:
  "trajectory_id": "{tid}",
  "prompt_messages": [list of {{"role": "system"|"user"|"assistant", "content": "..."}}],
  "tools": <use the tools list below exactly as-is>,
  "ground_truth_calls": [{{"name": "<tool_name>", "arguments": {{"<param>": <value>}}}}]
  OR ground_truth_calls: [] if the final request genuinely needs no tool call

CRITICAL GROUNDING RULES:
- Every argument VALUE must appear VERBATIM in the conversation (the user must have said it)
- Do NOT invent emails, IDs, timestamps, or specific values the user didn't state
- If a required value isn't in the conversation, write [] for ground_truth_calls
- Use realistic but simple values that a user would actually type

Tools available:
{tools_json}

Output only the JSON object:"""

    raw = _call(client, model, _SEED_GEN_SYSTEM, user_msg)
    if not raw:
        return None
    return _extract_json_obj(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Validation (mirrors evol_generate._validate / _check_grounding)
# ─────────────────────────────────────────────────────────────────────────────

KEEP_FIELDS = {"trajectory_id", "prompt_messages", "tools", "ground_truth_calls"}


def _tool_map(tools: list[dict]) -> dict[str, dict]:
    return {t["function"]["name"]: t["function"].get("parameters", {})
            for t in tools if "function" in t and "name" in t["function"]}


def _looks_injected(val: str) -> bool:
    if "@" in val:
        return True
    if re.search(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', val):
        return True
    if re.match(r'^[A-Za-z]{2,}[-_][A-Za-z0-9]', val):
        return True
    if re.search(r'[a-zA-Z][a-zA-Z0-9_-]*/[a-zA-Z0-9_-]', val):
        return True
    return False


def _check_grounding(gt_calls: list[dict], messages: list[dict]) -> list[str]:
    # Restrict to last user message + the assistant turn immediately before it
    current_context = ""
    found_last_user = False
    for m in reversed(messages):
        role = m.get("role", "")
        content = (m.get("content", "") or "").lower()
        current_context = content + " " + current_context
        if not found_last_user and role == "user":
            found_last_user = True
        elif found_last_user:
            for tc in m.get("tool_calls", []):
                current_context += " " + tc.get("function", {}).get("arguments", "").lower()
            if role == "assistant":
                break

    issues = []
    for call in gt_calls:
        for key, val in call.get("arguments", {}).items():
            if not isinstance(val, str) or len(val) < 4:
                continue
            if _looks_injected(val) and val.lower() not in current_context:
                issues.append(f"{call['name']}.{key}='{val}' not in conversation")
    return issues


def validate_seed(sample: dict) -> list[str]:
    reasons = []
    for f in KEEP_FIELDS:
        if f not in sample:
            reasons.append(f"missing field: {f}")
    if reasons:
        return reasons

    # Force tools to match domain tools (not whatever LLM returned)
    # (tools field is injected before this call, so just validate structure)
    tmap = _tool_map(sample["tools"])
    gt   = sample.get("ground_truth_calls", [])

    if not isinstance(gt, list):
        return ["ground_truth_calls is not a list"]

    for call in gt:
        name = call.get("name", "")
        if name and name not in tmap:
            reasons.append(f"GT tool '{name}' not in tools list")

    for call in gt:
        name = call.get("name", "")
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        for param in tmap.get(name, {}).get("required", []):
            if param not in args:
                reasons.append(f"GT '{name}' missing required param '{param}'")

    if not sample.get("prompt_messages"):
        reasons.append("empty prompt_messages")
    elif sample["prompt_messages"][0].get("role") != "system":
        reasons.append("first message is not system")

    if not reasons:
        grounding_issues = _check_grounding(gt, sample["prompt_messages"])
        if grounding_issues:
            reasons.extend(grounding_issues)

    return reasons


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(client, model, out_path: Path, seeds_per_domain: int, domains_filter: list[str] | None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_dropped = 0

    target_domains = domains_filter if domains_filter else list(DOMAINS.keys())

    for domain_key in target_domains:
        if domain_key not in DOMAINS:
            print(f"[WARN] Unknown domain '{domain_key}', skipping.")
            continue

        domain_info = DOMAINS[domain_key]
        print(f"\n{'='*60}")
        print(f"Domain: {domain_key}  ({domain_info['bfcl_category']})")
        print(f"{'='*60}")

        # Step A: generate tool schema
        print("  Step A — generating tool schema ...", flush=True)
        tools = None
        for attempt in range(3):
            tools = generate_tool_schema(client, model, domain_key, domain_info)
            if tools:
                break
            print(f"  [WARN] Tool schema attempt {attempt+1} failed, retrying...")
        if not tools:
            print(f"  [ERROR] Could not generate tool schema for {domain_key}, skipping domain.")
            continue
        print(f"  Generated {len(tools)} tools: {[t['function']['name'] for t in tools]}")

        # Step B: generate seed conversations
        domain_written = 0
        domain_dropped = 0

        for i in range(seeds_per_domain):
            sample = generate_seed_conversation(client, model, domain_key, tools, i)
            if sample is None:
                domain_dropped += 1
                continue

            # Override tools field with the canonical ones we generated
            sample["tools"] = tools

            reasons = validate_seed(sample)
            if reasons:
                domain_dropped += 1
                print(f"  [DROP {i:03d}] {reasons[0]}", flush=True)
                continue

            # Keep only 4 required fields
            clean = {k: sample[k] for k in KEEP_FIELDS if k in sample}

            with open(out_path, "a") as f:
                f.write(json.dumps(clean) + "\n")

            domain_written += 1
            if domain_written % 5 == 0 or domain_written == seeds_per_domain:
                print(f"  [{domain_written}/{seeds_per_domain}]", flush=True)

        total_written += domain_written
        total_dropped += domain_dropped
        print(f"  Domain done — written: {domain_written}, dropped: {domain_dropped}")

    print(f"\n{'='*60}")
    print(f"All domains done.")
    print(f"  Total written : {total_written}")
    print(f"  Total dropped : {total_dropped}")
    print(f"  Output file   : {out_path}")
    print(f"{'='*60}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Generate BFCL-aligned domain seed conversations.")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"Output JSONL file to APPEND seeds to (default: {DEFAULT_OUT})")
    ap.add_argument("--seeds-per-domain", type=int, default=SEEDS_PER_DOMAIN,
                    help=f"Seeds to generate per domain (default: {SEEDS_PER_DOMAIN})")
    ap.add_argument("--domains", default=None,
                    help="Comma-separated domain keys to generate (default: all 8)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=OPENROUTER_BASE)
    args = ap.parse_args()

    import os
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: provide --api-key or set OPENROUTER_API_KEY env var.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    domains_filter = None
    if args.domains:
        domains_filter = [d.strip() for d in args.domains.split(",")]

    run(
        client=client,
        model=args.model,
        out_path=Path(args.out),
        seeds_per_domain=args.seeds_per_domain,
        domains_filter=domains_filter,
    )


if __name__ == "__main__":
    main()
