from __future__ import annotations

import json
import re
import uuid
from typing import Any

from notionchat.exceptions import NotionChatError

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

_IDE_TOOL_EXACT = frozenset(
    {
        "read_file",
        "write",
        "write_file",
        "search_replace",
        "delete_file",
        "list_dir",
        "grep",
        "codebase_search",
        "run_terminal_cmd",
        "run_terminal_command",
        "edit_file",
        "create_file",
        "glob_file_search",
        "read_lints",
        "list_mcp_resources",
        "fetch_mcp_resource",
        "str_replace_editor",
        "read",
        "write",
        "strreplace",
        "shell",
        "delete",
        "semanticsearch",
        "glob",
        "readlints",
        "editnotebook",
        "task",
        "webfetch",
        "generateimage",
    }
)

_IDE_TOOL_SUBSTRINGS = (
    "file",
    "terminal",
    "directory",
    "codebase",
    "grep",
    "edit",
    "write",
    "mcp_",
    "notebook",
    "lint",
)

_DENIAL_PHRASES = (
    "i'm notion ai",
    "i am notion ai",
    "i'm claude",
    "i am claude",
    "i'm not cursor",
    "i am not cursor",
    "prompt injection",
    "can't directly create",
    "cannot directly create",
    "don't have access to your local",
    "do not have access to your local",
    "cannot write directly",
    "can't write directly",
    "don't have access to your cursor",
    "cannot access your workspace",
    "can't access your workspace",
    "paste-ready code",
    "paste your project structure",
    "despite what the embedded context claims",
    "can't directly create or edit",
    "cannot directly create or edit",
    "don't have access to cursor",
    "do not have access to cursor",
    "copy-pasteable",
    "paste either",
    "can't directly operate",
    "cannot directly operate",
    "don't have access to your g:",
    "glob or read first",
    "don't have access to cursor's",
    "do not have access to cursor's",
    "read/write/shell tools",
    "this chat environment",
    "can't directly create or edit your",
    "cannot directly create or edit your",
    "local workspace filesystem",
    "ready-to-paste",
    "file contents to paste",
    "tell me which you prefer",
)

_CURSOR_FALLBACK_TOOL_NAMES = (
    "Glob",
    "Read",
    "Write",
    "StrReplace",
    "Shell",
    "Grep",
    "SemanticSearch",
    "Delete",
    "ReadLints",
    "list_dir",
    "run_terminal_cmd",
)

_CODING_TASK_HINTS = (
    "create",
    "build",
    "scaffold",
    "implement",
    "add ",
    "fix ",
    "write ",
    "generate ",
    "app",
    "component",
    "page",
    "project",
    "vite",
    "react",
    "tailwind",
    "shadcn",
    "typescript",
    "coffee",
)

_TOOL_NAME_ALIASES: dict[str, str] = {
    "read_file": "Read",
    "read": "Read",
    "write_file": "Write",
    "write": "Write",
    "search_replace": "StrReplace",
    "str_replace_editor": "StrReplace",
    "strreplace": "StrReplace",
    "run_terminal_cmd": "Shell",
    "run_terminal_command": "Shell",
    "list_dir": "Glob",
    "glob_file_search": "Glob",
    "codebase_search": "SemanticSearch",
    "grep": "Grep",
    "delete_file": "Delete",
}


def _new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _extract_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type == "text":
                parts.append(str(item.get("text", "")))
            elif item_type in ("tool_result", "tool_use_result", "tool_result_error"):
                parts.append(str(item.get("content") or item.get("output") or item.get("text") or ""))
            elif "text" in item:
                parts.append(str(item["text"]))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts)


def _message_content_has_tool_result(content: str | list[Any] | None) -> bool:
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") in (
            "tool_result",
            "tool_use_result",
            "tool_result_error",
        ):
            return True
    return False


def normalize_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not tools:
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            out.append(tool)
            continue
        if "name" in tool:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
    return out


def _tool_choice_instruction(tool_choice: str | dict[str, Any] | None) -> str:
    if tool_choice is None or tool_choice == "auto":
        return "Call a tool only when it helps answer the user."
    if tool_choice == "none":
        return ""
    if tool_choice == "required":
        return "You MUST call at least one tool before answering."
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(name, str) and name:
            return f"You MUST call the `{name}` tool."
    return "Call a tool only when it helps answer the user."


def is_ide_agent_tools(tools: list[dict[str, Any]] | None) -> bool:
    for tool in normalize_tools(tools):
        name = str((tool.get("function") or {}).get("name", "")).lower()
        if not name:
            continue
        if name in _IDE_TOOL_EXACT:
            return True
        if any(part in name for part in _IDE_TOOL_SUBSTRINGS):
            return True
    return False


def client_tool_names(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in normalize_tools(tools):
        name = (tool.get("function") or {}).get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names


_LOOP_EXPLORE_TOOLS = frozenset(
    {
        "Glob",
        "Grep",
        "SemanticSearch",
        "codebase_search",
        "glob_file_search",
        "list_dir",
    }
)

_ACTION_TOOLS = frozenset(
    {
        "Write",
        "write",
        "write_file",
        "StrReplace",
        "search_replace",
        "str_replace_editor",
        "Shell",
        "run_terminal_cmd",
        "run_terminal_command",
        "Delete",
        "delete_file",
        "EditNotebook",
    }
)

_LANG_DEFAULT_PATH: dict[str, str] = {
    "html": "index.html",
    "tsx": "src/App.tsx",
    "typescript": "src/App.tsx",
    "jsx": "src/App.jsx",
    "javascript": "src/App.jsx",
    "js": "src/App.jsx",
    "css": "src/index.css",
    "json": "package.json",
    "ts": "src/main.ts",
}


def align_tool_calls_to_client(
    tool_calls: list[dict[str, Any]],
    client_tools: list[dict[str, Any]] | None,
    *,
    allow_aliases: bool = True,
) -> list[dict[str, Any]]:
    allowed = client_tool_names(client_tools)
    if not allowed:
        return normalize_tool_calls(tool_calls)

    lower_to_canonical = {name.lower(): name for name in allowed}
    if allow_aliases:
        for alias, target in _TOOL_NAME_ALIASES.items():
            if target in allowed:
                lower_to_canonical[alias.lower()] = target

    out: list[dict[str, Any]] = []
    for tc in normalize_tool_calls(tool_calls):
        name = str((tc.get("function") or {}).get("name", ""))
        if name in allowed:
            out.append(tc)
            continue
        if allow_aliases:
            mapped = lower_to_canonical.get(name.lower())
            if mapped:
                fixed = dict(tc)
                fn = dict(tc.get("function") or {})
                fn["name"] = mapped
                fixed["function"] = fn
                out.append(fixed)
    return out


def filter_agent_tool_calls(
    tool_calls: list[dict[str, Any]],
    *,
    prompt: str,
    ide_agent: bool = False,
    messages: list[Any] | None = None,
) -> list[dict[str, Any]]:
    if not tool_calls:
        return []
    has_history = prompt_has_tool_history(prompt) or (
        bool(messages) and conversation_has_tool_history(messages)
    )
    if not (ide_agent and has_history):
        return tool_calls

    last_names = last_assistant_tool_names(messages or [])
    last_loop = [n for n in last_names if n in _LOOP_EXPLORE_TOOLS]
    if not last_loop:
        return tool_calls

    blocked = set(last_loop)
    return [
        tc
        for tc in tool_calls
        if str((tc.get("function") or {}).get("name", "")) not in blocked
    ]


def looks_like_tool_denial(text: str | None) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _DENIAL_PHRASES)


def looks_like_coding_task_text(text: str) -> bool:
    lower = text.lower()
    return any(hint in lower for hint in _CODING_TASK_HINTS)


def looks_like_coding_task_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    tail = lower.rsplit("user:", 1)[-1] if "user:" in lower else lower
    return looks_like_coding_task_text(tail)


def infer_scaffold_command(user_request: str) -> str | None:
    lower = user_request.lower()
    if "next.js" in lower or "nextjs" in lower or re.search(r"\bnext\b", lower):
        return "npm create next-app@latest . -- --typescript --tailwind --eslint --app --no-src-dir --import-alias '@/*'"
    if any(token in lower for token in ("vite", "react", "tailwind", "shadcn", "tsx", "typescript")):
        return "npm create vite@latest . -- --template react-ts"
    if looks_like_coding_task_text(user_request):
        return "npm create vite@latest . -- --template react-ts"
    return None


def should_bootstrap_scaffold(messages: list[Any], notion_text: str | None) -> bool:
    if conversation_has_scaffold_tool_result(messages):
        return False
    if conversation_had_scaffold_shell(messages):
        return False

    user_request = extract_last_user_request(messages)
    if not infer_scaffold_command(user_request):
        return False

    if not conversation_has_tool_history(messages):
        return True
    if notion_text and looks_like_tool_denial(notion_text):
        return True
    return False


def bootstrap_agent_tool_calls(
    *,
    messages: list[Any],
    notion_text: str | None,
    client_tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """When Notion refuses or returns no compilable tools, start scaffold Shell ourselves."""
    if not should_bootstrap_scaffold(messages, notion_text):
        return []

    shell_tool = _pick_tool(client_tools, "Shell", "run_terminal_cmd", "run_terminal_command")
    if not shell_tool:
        return []

    command = infer_scaffold_command(extract_last_user_request(messages))
    if not command:
        return []

    command = normalize_scaffold_command(command)
    tc = normalize_shell_tool_call(
        _make_tool_call(
            shell_tool,
            {
                "command": command,
                "description": "Scaffold project in current workspace",
                "block_until_ms": _SCAFFOLD_BLOCK_MS,
            },
        )
    )
    return [tc]


def cursor_fallback_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }
        for name in _CURSOR_FALLBACK_TOOL_NAMES
    ]


def is_ide_agent_messages(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "system":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = _extract_text(content).lower()
        if not text:
            continue
        agent_markers = (
            "cursor",
            "composer",
            "coding assistant",
            "you are pair programming",
            "tool_calls",
            "function calling",
        )
        tool_markers = (
            "read",
            "glob",
            "strreplace",
            "run_terminal",
            "codebase_search",
            "search_replace",
            "list_dir",
        )
        if any(m in text for m in agent_markers) and any(t in text for t in tool_markers):
            return True
    return False


def prompt_has_tool_history(prompt: str) -> bool:
    return "Tool `" in prompt or "Assistant: [tool call `" in prompt


def conversation_has_tool_history(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "tool":
            return True
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        if role == "user" and _message_content_has_tool_result(content):
            return True
        if role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls is None and isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
            if tool_calls:
                return True
    return False


_CODE_FENCE_RE = re.compile(
    r"```(?:(\w+)?(?::([^\n]+))?)?\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_FILE_IN_REQUEST_RE = re.compile(
    r"[`\"']?([\w./\\-]+\.(?:html?|tsx|ts|jsx|js|css|json|md|py|vue|txt))[`\"']?",
    re.IGNORECASE,
)


def _pick_tool(client_tools: list[dict[str, Any]] | None, *candidates: str) -> str | None:
    allowed = client_tool_names(client_tools)
    for name in candidates:
        if name in allowed:
            return name
    return None


def extract_last_user_request(messages: list[Any]) -> str:
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        text = _extract_text(content).strip()
        if text:
            return text
    return ""


def infer_file_path(
    user_request: str,
    *,
    lang: str | None = None,
    path_hint: str | None = None,
) -> str | None:
    if path_hint:
        return path_hint.replace("\\", "/")
    match = _FILE_IN_REQUEST_RE.search(user_request)
    if match:
        return match.group(1).replace("\\", "/")
    lower_req = user_request.lower()
    if lang:
        lang_l = lang.lower()
        if lang_l in ("ts", "typescript") and "vite" in lower_req:
            return "vite.config.ts"
        if lang_l in ("js", "javascript") and "vite" in lower_req:
            return "vite.config.js"
        path = _LANG_DEFAULT_PATH.get(lang_l)
        if path:
            return path
    return None


def extract_code_fences(text: str) -> list[tuple[str | None, str | None, str]]:
    blocks: list[tuple[str | None, str | None, str]] = []
    for match in _CODE_FENCE_RE.finditer(text):
        lang = (match.group(1) or "").strip().lower() or None
        path_hint = (match.group(2) or "").strip() or None
        code = (match.group(3) or "").strip()
        if code:
            blocks.append((lang, path_hint, code))
    return blocks


_SHELL_FENCE_RE = re.compile(
    r"```(?:bash|sh|shell|zsh|powershell|terminal|cmd)?\s*\n([\s\S]*?)```",
    re.IGNORECASE,
)

_CMD_LINE_RE = re.compile(
    r"^(?:npm|npx|pnpm|yarn|bun|cd|mkdir|curl|git)\b.+$",
    re.MULTILINE | re.IGNORECASE,
)

_PACKAGE_MANAGER_CREATE_RE = re.compile(
    r"^(npm|pnpm|yarn|bun)\s+create\s+(\S+)(?:\s+(\S+))?(\s+--.*)?$",
    re.IGNORECASE,
)
_NPX_CREATE_RE = re.compile(
    r"^(npx)\s+((?:create-|@)[^\s]+)(?:\s+(\S+))?(\s+--.*)?$",
    re.IGNORECASE,
)

_SHELL_TOOL_NAMES = frozenset({"Shell", "run_terminal_cmd", "run_terminal_command"})
_SCAFFOLD_BLOCK_MS = 120_000


def is_scaffold_command(command: str) -> bool:
    lower = command.strip().lower()
    if re.match(r"^(npm|pnpm|yarn|bun)\s+create\b", lower):
        return True
    return bool(re.match(r"^npx\s+create-", lower))


def normalize_scaffold_command(command: str) -> str:
    """Force npm/pnpm create-* to scaffold in the current workspace (`.`), not a subfolder."""
    cmd = command.strip()
    if not is_scaffold_command(cmd):
        return cmd

    match = _PACKAGE_MANAGER_CREATE_RE.match(cmd)
    if match:
        pm, pkg, target, flags = match.group(1), match.group(2), match.group(3), match.group(4) or ""
        if target is None or target.startswith("-"):
            return f"{pm} create {pkg} .{flags}" if flags else f"{pm} create {pkg} ."
        if target != ".":
            return f"{pm} create {pkg} .{flags}"
        return cmd

    match = _NPX_CREATE_RE.match(cmd)
    if match:
        npx, pkg, target, flags = match.group(1), match.group(2), match.group(3), match.group(4) or ""
        if target is None or target.startswith("-"):
            return f"{npx} {pkg} .{flags}" if flags else f"{npx} {pkg} ."
        if target != ".":
            return f"{npx} {pkg} .{flags}"
        return cmd

    return cmd


def _shell_command_from_tool_call(tc: dict[str, Any]) -> str | None:
    name = str((tc.get("function") or {}).get("name", ""))
    if name not in _SHELL_TOOL_NAMES:
        return None
    raw_args = (tc.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError):
        return None
    command = args.get("command")
    return command if isinstance(command, str) else None


def normalize_shell_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    command = _shell_command_from_tool_call(tc)
    if not command:
        return tc

    normalized = normalize_scaffold_command(command)
    if normalized == command and not is_scaffold_command(command):
        return tc

    raw_args = (tc.get("function") or {}).get("arguments", "{}")
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError):
        args = {"command": normalized}
    else:
        args = dict(args)
        args["command"] = normalized

    if is_scaffold_command(normalized):
        args.setdefault("description", normalized[:120])
        args["block_until_ms"] = max(int(args.get("block_until_ms") or 0), _SCAFFOLD_BLOCK_MS)

    fixed = dict(tc)
    fn = dict(tc.get("function") or {})
    fn["arguments"] = json.dumps(args, ensure_ascii=False)
    fixed["function"] = fn
    return fixed


def conversation_had_scaffold_shell(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "assistant":
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        for tc in tool_calls or []:
            if isinstance(tc, dict):
                command = _shell_command_from_tool_call(tc)
            else:
                fn = tc.function
                try:
                    args = json.loads(fn.arguments)
                    command = args.get("command")
                except (json.JSONDecodeError, AttributeError, TypeError):
                    command = None
            if isinstance(command, str) and is_scaffold_command(normalize_scaffold_command(command)):
                return True
    return False


def conversation_has_scaffold_tool_result(messages: list[Any]) -> bool:
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if role == "tool":
            name = getattr(msg, "name", None) or (msg.get("name") if isinstance(msg, dict) else None)
            text = _extract_text(content).lower()
            if str(name or "").lower() in {n.lower() for n in _SHELL_TOOL_NAMES}:
                return True
            if any(marker in text for marker in ("exit code", "npm create", "pnpm create", "npx create")):
                return True

        if role == "user" and _message_content_has_tool_result(content):
            text = _extract_text(content).lower()
            if "tool `" in text and any(
                marker in text for marker in ("shell", "npm", "pnpm", "npx create", "exit code")
            ):
                return True
    return False


def sequentialize_agent_tool_calls(
    tool_calls: list[dict[str, Any]],
    messages: list[Any],
) -> list[dict[str, Any]]:
    """One scaffold Shell per turn; never parallel Shell scaffold + Write."""
    if not tool_calls:
        return []

    normalized = [normalize_shell_tool_call(tc) for tc in tool_calls]
    scaffold_done = conversation_had_scaffold_shell(messages) and conversation_has_scaffold_tool_result(
        messages
    )

    scaffold_shells: list[dict[str, Any]] = []
    other_calls: list[dict[str, Any]] = []
    for tc in normalized:
        command = _shell_command_from_tool_call(tc)
        name = str((tc.get("function") or {}).get("name", ""))
        if name in _SHELL_TOOL_NAMES and command and is_scaffold_command(command):
            if not scaffold_done:
                scaffold_shells.append(tc)
            continue
        other_calls.append(tc)

    if scaffold_shells:
        return scaffold_shells[:1]

    return other_calls


def _make_tool_call(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _new_tool_call_id(),
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _make_write_tool_call(tool_name: str, path: str, contents: str) -> dict[str, Any]:
    return _make_tool_call(
        tool_name,
        {"path": path.replace("\\", "/"), "contents": contents},
    )


def last_assistant_tool_names(messages: list[Any]) -> list[str]:
    names: list[str] = []
    for msg in reversed(messages):
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role != "assistant":
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        if not tool_calls:
            break
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                name = fn.get("name")
            else:
                name = tc.function.name
            if isinstance(name, str) and name:
                names.append(name)
        break
    return names


def dedupe_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for tc in normalize_tool_calls(tool_calls):
        fn = tc.get("function") or {}
        key = f"{fn.get('name')}::{fn.get('arguments')}"
        if key in seen:
            continue
        seen.add(key)
        out.append(tc)
    return out


def extract_shell_commands(text: str) -> list[str]:
    commands: list[str] = []
    for match in _SHELL_FENCE_RE.finditer(text):
        block = match.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if _CMD_LINE_RE.match(line):
                commands.append(line)
    for match in _CMD_LINE_RE.finditer(text):
        commands.append(match.group(0).strip())
    seen: set[str] = set()
    out: list[str] = []
    for cmd in commands:
        if cmd not in seen:
            seen.add(cmd)
            out.append(cmd)
    return out


def extract_all_tool_calls_from_text(text: str) -> list[dict[str, Any]]:
    if not text.strip():
        return []
    obj = _try_parse_json_object(text)
    if obj and obj.get("tool_calls"):
        return normalize_tool_calls(obj.get("tool_calls"))
    found: list[dict[str, Any]] = []
    for match in re.finditer(r'"tool_calls"\s*:\s*\[', text):
        brace = text.rfind("{", 0, match.start())
        if brace < 0:
            continue
        depth = 0
        for i in range(brace, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        chunk = json.loads(text[brace : i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(chunk, dict) and chunk.get("tool_calls"):
                        found.extend(normalize_tool_calls(chunk.get("tool_calls")))
                    break
    return dedupe_tool_calls(found)


def synthesize_shell_tool_calls(
    *,
    notion_text: str,
    client_tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    shell_tool = _pick_tool(client_tools, "Shell", "run_terminal_cmd", "run_terminal_command")
    if not shell_tool:
        return []
    commands = [
        normalize_scaffold_command(cmd)
        for cmd in extract_shell_commands(notion_text)
    ]
    if not commands:
        return []
    out: list[dict[str, Any]] = []
    for cmd in commands[:3]:
        args: dict[str, Any] = {"command": cmd, "description": cmd[:120]}
        if is_scaffold_command(cmd):
            args["block_until_ms"] = _SCAFFOLD_BLOCK_MS
        out.append(_make_tool_call(shell_tool, args))
    return out


def synthesize_writes_from_fences(
    *,
    messages: list[Any],
    notion_text: str,
    client_tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Only synthesize Write from explicit fenced code blocks — never guess HTML apps."""
    write_tool = _pick_tool(client_tools, "Write", "write", "write_file")
    if not write_tool:
        return []

    user_request = extract_last_user_request(messages)
    writes: list[dict[str, Any]] = []
    for lang, path_hint, code in extract_code_fences(notion_text):
        path = infer_file_path(user_request, lang=lang, path_hint=path_hint)
        if not path:
            continue
        writes.append(_make_write_tool_call(write_tool, path, code))

    return writes


def compile_agent_tool_calls(
    *,
    messages: list[Any],
    notion_text: str | None,
    notion_tool_calls: list[dict[str, Any]] | None,
    client_tools: list[dict[str, Any]] | None,
    prompt: str = "",
) -> tuple[str | None, list[dict[str, Any]]]:
    """Compile Notion output into Cursor-compatible tool_calls (multi-tool, multi-source)."""
    text = notion_text or ""
    collected: list[dict[str, Any]] = []

    for tc in extract_all_tool_calls_from_text(text):
        collected.append(tc)

    content, parsed = parse_assistant_output(text)
    if parsed:
        collected.extend(parsed)

    if notion_tool_calls:
        collected.extend(
            align_tool_calls_to_client(
                notion_tool_calls,
                client_tools,
                allow_aliases=True,
            )
        )

    collected = dedupe_tool_calls(collected)
    collected = sequentialize_agent_tool_calls(collected, messages)
    collected = filter_agent_tool_calls(
        collected,
        prompt=prompt,
        ide_agent=True,
        messages=messages,
    )
    if collected:
        return content, collected

    shells = synthesize_shell_tool_calls(notion_text=text, client_tools=client_tools)
    writes = synthesize_writes_from_fences(
        messages=messages,
        notion_text=text,
        client_tools=client_tools,
    )
    compiled = dedupe_tool_calls(shells + writes)
    compiled = sequentialize_agent_tool_calls(compiled, messages)
    if compiled:
        return None, compiled

    bootstrapped = bootstrap_agent_tool_calls(
        messages=messages,
        notion_text=text,
        client_tools=client_tools,
    )
    if bootstrapped:
        return None, bootstrapped

    if looks_like_tool_denial(text):
        return None, []

    return notion_text, []


def bridge_ide_agent_response(
    *,
    messages: list[Any],
    notion_text: str | None,
    notion_tool_calls: list[dict[str, Any]] | None,
    client_tools: list[dict[str, Any]] | None,
    prompt: str = "",
) -> tuple[str | None, list[dict[str, Any]]]:
    return compile_agent_tool_calls(
        messages=messages,
        notion_text=notion_text,
        notion_tool_calls=notion_tool_calls,
        client_tools=client_tools,
        prompt=prompt,
    )


def build_ide_tools_instruction(
    tools: list[dict[str, Any]],
    *,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    normalized = normalize_tools(tools)
    names = [
        str((t.get("function") or {}).get("name", ""))
        for t in normalized
    ]
    names = [n for n in names if n]
    specs = json.dumps(normalized, ensure_ascii=False, indent=2)
    if len(specs) > 14000:
        specs = specs[:14000] + "\n... (truncated)"
    choice = _tool_choice_instruction(tool_choice)
    if tool_choice in (None, "auto"):
        choice = (
            "You are the coding agent inside Cursor IDE. Tools execute on the user's machine.\n"
            "Scaffold in the CURRENT workspace folder: use `npm create vite@latest . -- --template react-ts` "
            "(`.` not a subfolder name). Run scaffold Shell alone first; wait for tool result before Write/StrReplace.\n"
            "Never emit Shell scaffold and Write in the same response."
        )
    return (
        "OpenAI function-calling channel for Cursor IDE.\n"
        f"{choice}\n\n"
        f"Callable tools (exact names): {', '.join(names)}\n\n"
        "Tool schemas (JSON):\n"
        f"{specs}\n\n"
        "When calling tools, respond with ONLY JSON (no markdown fences around JSON):\n"
        '{"content": null, "tool_calls": [{"id": "call_<unique>", "type": "function", '
        '"function": {"name": "Write", "arguments": "{\\"path\\":\\"src/App.tsx\\",\\"contents\\":\\"...\\"}"}}]}\n\n'
        "For multiple new files you may also use path-tagged fences:\n"
        "```tsx:src/App.tsx\n...full file...\n```\n"
        "For shell setup use JSON Shell tool_calls or ```bash\\nnpm create vite@latest . -- --template react-ts\\n```\n"
        "Never say you lack filesystem access. Never reply with only terminal instructions — call tools."
    )


def build_tool_denial_retry_append() -> str:
    return (
        "Use `npm create vite@latest .` in the current folder (not a new subfolder). "
        "Call Shell scaffold alone first; after it finishes, call Write for files. "
        "JSON tool_calls or ```tsx:src/App.tsx\\n...\\n``` path-tagged fences."
    )


def build_tools_system_append(
    tools: list[dict[str, Any]],
    *,
    tool_choice: str | dict[str, Any] | None = None,
    ide_agent: bool = False,
) -> str:
    if not tools or tool_choice == "none":
        return ""
    if ide_agent:
        return build_ide_tools_instruction(tools, tool_choice=tool_choice)
    choice_line = _tool_choice_instruction(tool_choice)
    specs = json.dumps(tools, ensure_ascii=False, indent=2)
    return (
        "You are an assistant that can call external tools using OpenAI function calling.\n"
        f"{choice_line}\n\n"
        "Available tools (JSON Schema):\n"
        f"{specs}\n\n"
        "When you need to call one or more tools, respond with ONLY valid JSON (no markdown fences):\n"
        '{"content": null, "tool_calls": [{"id": "call_<unique>", "type": "function", '
        '"function": {"name": "<tool_name>", "arguments": "<JSON string>"}}]}\n\n'
        "When you do not need tools, respond with normal plain text, OR JSON:\n"
        '{"content": "<your reply>", "tool_calls": []}\n\n'
        "Rules:\n"
        "- `arguments` must be a JSON string (escaped), not a raw object.\n"
        "- Use exact tool names from the list above.\n"
        "- Each tool call needs a unique `id` starting with `call_`.\n"
    )


def _format_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
    name = fn.get("name") or tc.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("tool call missing function name")
    raw_args = fn.get("arguments", tc.get("arguments", "{}"))
    if isinstance(raw_args, dict):
        args = json.dumps(raw_args, ensure_ascii=False)
    elif isinstance(raw_args, str):
        args = raw_args
    else:
        args = "{}"
    return {
        "id": str(tc.get("id") or _new_tool_call_id()),
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def normalize_tool_calls(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            out.append(_format_tool_call(item))
        except ValueError:
            continue
    return out


def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            obj = json.loads(match.group(1))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        return None
    if '"tool_calls"' in text:
        idx = text.find('"tool_calls"')
        brace = text.rfind("{", 0, idx)
        if brace >= 0:
            depth = 0
            for i in range(brace, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(text[brace : i + 1])
                            return obj if isinstance(obj, dict) else None
                        except json.JSONDecodeError:
                            break
    return None


def parse_assistant_output(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """Parse model text into (content, tool_calls). Plain text => (text, [])."""
    stripped = text.strip()
    if not stripped:
        return None, []

    obj = _try_parse_json_object(stripped)
    if obj is not None:
        tool_calls = normalize_tool_calls(obj.get("tool_calls"))
        if tool_calls:
            content = obj.get("content")
            if content is None:
                return None, tool_calls
            if isinstance(content, str):
                return content or None, tool_calls
            return json.dumps(content, ensure_ascii=False), tool_calls
        content = obj.get("content")
        if isinstance(content, str) and content:
            return content, []
        if content is not None and not tool_calls:
            return json.dumps(content, ensure_ascii=False), []

    return stripped, []


def prepare_chat_input(
    messages: list[Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> tuple[str | None, str, bool, bool, list[dict[str, Any]]]:
    """Build (system, prompt, tools_active, ide_agent, tools) for Notion from OpenAI messages."""
    cursor_ide = is_ide_agent_messages(messages)
    normalized_tools = normalize_tools(tools)
    if not normalized_tools and cursor_ide:
        normalized_tools = cursor_fallback_tools()
    tools_active = bool(normalized_tools) and tool_choice != "none"
    ide_agent = tools_active and (is_ide_agent_tools(normalized_tools) or cursor_ide)

    system_parts: list[str] = [
        "You are a helpful assistant. Respond directly in the conversation. "
        "Do not create, draft, or render Notion pages. "
        "Answer inline in the conversation."
    ]
    transcript_blocks: list[str] = []
    pending_tool_results: list[str] = []

    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if not role:
            continue

        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")

        if role == "system":
            text = _extract_text(content).strip()
            if text:
                system_parts.append(text)
        elif role == "user":
            if pending_tool_results:
                transcript_blocks.extend(pending_tool_results)
                pending_tool_results = []
            if isinstance(content, list):
                user_text_parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        user_text_parts.append(str(item.get("text", "")))
                    elif item_type in ("tool_result", "tool_use_result", "tool_result_error"):
                        label = (
                            item.get("tool_name")
                            or item.get("name")
                            or item.get("tool_use_id")
                            or "tool"
                        )
                        output = item.get("content") or item.get("output") or item.get("text") or ""
                        pending_tool_results.append(f"Tool `{label}` result:\n{output}")
                text = "\n".join(user_text_parts).strip()
            else:
                text = _extract_text(content).strip()
            if text:
                transcript_blocks.append(f"User: {text}")
        elif role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls is None and isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn = tc.get("function") or {}
                        name = fn.get("name", "unknown")
                        args = fn.get("arguments", "{}")
                    else:
                        fn = tc.function
                        name = fn.name
                        args = fn.arguments
                    transcript_blocks.append(
                        f"Assistant: [tool call `{name}` args={args}]"
                    )
            text = _extract_text(content).strip()
            if text:
                transcript_blocks.append(f"Assistant: {text}")
        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id is None and isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id")
            name = getattr(msg, "name", None)
            if name is None and isinstance(msg, dict):
                name = msg.get("name")
            label = name or tool_call_id or "tool"
            result = _extract_text(content).strip()
            pending_tool_results.append(f"Tool `{label}` result:\n{result}")

    if pending_tool_results:
        transcript_blocks.extend(pending_tool_results)
        transcript_blocks.append("User: Continue using the tool results above.")

    if not transcript_blocks:
        raise NotionChatError("No user message in request", status_code=400)

    if tools_active:
        system_parts.append(
            build_tools_system_append(
                normalized_tools,
                tool_choice=tool_choice,
                ide_agent=ide_agent,
            )
        )

    system = "\n\n".join(system_parts) if system_parts else None
    prompt = "\n\n".join(transcript_blocks)
    return system, prompt, tools_active, ide_agent, normalized_tools


def merge_tool_calls(
    *,
    text: str,
    ndjson_tool_calls: list[dict[str, Any]],
    tools_active: bool,
    client_tools: list[dict[str, Any]] | None = None,
    prompt: str = "",
    ide_agent: bool = False,
    messages: list[Any] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    content, parsed = parse_assistant_output(text) if tools_active else (text, [])
    if parsed:
        aligned = align_tool_calls_to_client(parsed, client_tools, allow_aliases=True)
        aligned = filter_agent_tool_calls(
            aligned,
            prompt=prompt,
            ide_agent=ide_agent,
            messages=messages,
        )
        if aligned:
            return content, aligned
    if ndjson_tool_calls:
        aligned = align_tool_calls_to_client(
            ndjson_tool_calls,
            client_tools,
            allow_aliases=ide_agent,
        )
        aligned = filter_agent_tool_calls(
            aligned,
            prompt=prompt,
            ide_agent=ide_agent,
            messages=messages,
        )
        if aligned:
            return (content if content else None), aligned
    return (content if content else text or None), []
