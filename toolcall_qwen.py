import json

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from overrides import override

_TOOL_SYSTEM_PREFIX = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)
_TOOL_SYSTEM_SUFFIX = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "**Important Notes**\n"
    "1. When an appropriate tool is available, you MUST use a tool call — "
    "do NOT answer with plain text, math, or code.\n"
    "2. Include optional parameters when the user explicitly specifies them "
    "(e.g. units, return format, method).\n"
    "3. Use correct data types: integers must be numbers not strings, "
    "arrays must be lists.\n"
    "4. If no tool applies or required parameters are missing, state this "
    "directly without making a tool call."
)


class ToolCallQwenHandler(OSSHandler):
    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # Skip base-class system_prompt_pre_processing_chat_model — it injects
        # a BFCL-format system prompt that conflicts with ours in _format_prompt.
        return {"message": [], "function": test_entry["function"]}

    @override
    def _format_prompt(self, messages, function):
        # Wrap bare BFCL function dicts in OpenAI tool format to match training.
        tool_list = [
            self._to_openai_tool(t)
            for t in (function if isinstance(function, list) else [function])
        ]
        tools_str = "".join("\n" + json.dumps(t) for t in tool_list)
        tool_block = f"{_TOOL_SYSTEM_PREFIX}{tools_str}{_TOOL_SYSTEM_SUFFIX}"

        formatted_prompt = ""
        start_idx = 0

        if messages and messages[0]["role"] == "system":
            sys_content = messages[0]["content"]
            formatted_prompt += (
                f"<|im_start|>system\n{sys_content}\n\n{tool_block}<|im_end|>\n"
            )
            start_idx = 1
        else:
            formatted_prompt += f"<|im_start|>system\n{tool_block}<|im_end|>\n"

        for idx, message in enumerate(messages[start_idx:], start=start_idx):
            role = message["role"]
            content = message.get("content") or ""

            if role == "user":
                formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"

            elif role == "assistant":
                parts = []
                if content:
                    parts.append(content)
                for tc in message.get("tool_calls", []):
                    func = tc.get("function", tc)
                    name = func.get("name", "")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    parts.append(
                        f"<tool_call>\n"
                        f'{{\"name\": \"{name}\", \"arguments\": {json.dumps(args)}}}\n'
                        f"</tool_call>"
                    )
                formatted_prompt += (
                    f"<|im_start|>assistant\n" + "\n".join(parts) + "<|im_end|>\n"
                )

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = (
                    messages[idx + 1]["role"] if idx < len(messages) - 1 else None
                )
                if prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"
                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"
                if next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def decode_ast(self, result, language="Python", has_tool_call_tag=True):
        return [
            {obj["name"]: obj["arguments"]}
            for obj in self._extract_tool_calls(result)
            if obj.get("name")
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag=True):
        calls = []
        for obj in self._extract_tool_calls(result):
            name = obj.get("name", "")
            if not name:
                continue
            args_str = ", ".join(
                f"{k}={repr(v)}" for k, v in obj.get("arguments", {}).items()
            )
            calls.append(f"{name}({args_str})")
        return calls

    @staticmethod
    def _to_openai_tool(func: dict) -> dict:
        """Wrap a bare BFCL function dict in OpenAI tool format to match training."""
        if "type" in func:
            return func
        return {"type": "function", "function": func}

    @staticmethod
    def _extract_tool_calls(result):
        calls = []
        for part in result.split("<tool_call>")[1:]:
            raw = part.split("</tool_call>")[0].strip()
            # Strip up to 3 trailing '}' to recover from the model's }}} bug.
            for candidate in [raw, raw[:-1], raw[:-2], raw[:-3]]:
                try:
                    obj = json.loads(candidate)
                    calls.append(obj)
                    break
                except json.JSONDecodeError:
                    continue
        return calls
