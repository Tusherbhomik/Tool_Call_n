"""
utils.py — Shared helpers used by reward.py and data_loader.py.

Keeps tool-call parsing logic in one place so reward + audit scripts agree.
"""
import re
import json


def parse_tool_calls(text: str) -> list[dict]:
    """
    Parse tool calls from model output.

    Handles two formats:
      1. <tool_call>{"name": "...", "arguments": {...}}</tool_call>
      2. Bare JSON: {"name": "...", "arguments": {...}}  (fallback)

    Returns a list of dicts, each with keys "name" and "arguments" (always a dict).
    Malformed JSON is silently skipped.
    """
    calls = []

    # Primary format: <tool_call> tags
    for block in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue

        name = obj.get("name", "")
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if name:
            calls.append({
                "name": name,
                "arguments": args if isinstance(args, dict) else {},
            })

    # Fallback: bare JSON tool calls
    if not calls:
        i = 0
        while i < len(text):
            # Find next potential JSON object starting with {"name"
            m = re.search(r'\{"name"\s*:\s*"([^"]+)"', text[i:])
            if not m:
                break
            start = i + m.start()

            # Walk forward to find matching closing brace, respecting nesting
            depth = 0
            end = -1
            in_string = False
            escape = False
            for j in range(start, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break

            if end == -1:
                break

            try:
                obj = json.loads(text[start:end])
                name = obj.get("name", "")
                args = obj.get("arguments", obj.get("parameters", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if name:
                    calls.append({
                        "name": name,
                        "arguments": args if isinstance(args, dict) else {},
                    })
            except json.JSONDecodeError:
                pass

            i = end

    return calls


def norm_value(v) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if v is None:
        return ""
    if isinstance(v, list):
        return json.dumps(sorted(v, key=str) if all(isinstance(x, (str, int, float)) for x in v) else v, sort_keys=True)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v).strip().lower()
