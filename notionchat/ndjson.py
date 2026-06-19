from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)

_SEARCH_PREAMBLE_MARKERS = (
    "let me search",
    "let me look up",
    "let me look into",
    "let me check",
    "let me find",
    "let me get the latest",
    "let me verify",
    "let me gather",
    "i'll search",
    "i will search",
    "i'll look up",
    "i will look up",
    "searching for",
    "looking up",
    "checking the latest",
)

_NOTION_PAGE_PREAMBLE_MARKERS = (
    "i'll create this as a notion page",
    "i will create this as a notion page",
    "i'll create a notion page",
    "i will create a notion page",
    "create this as a notion page",
    "create a notion page",
    "i'll write this as a notion page",
    "i will write this as a notion page",
    "let me create a notion page",
    "let me write this as a notion page",
    "i'll draft a notion page",
    "i will draft a notion page",
    "as a notion page so you have",
    "as a notion page for you",
    "clean, formatted long-form article",
    "i'll create this as a page",
    "i will create this as a page",
    "let me write it as a page",
    "i'll render this as a notion page",
    "i will render this as a notion page",
    "i'll save this as a notion page",
    "i will save this as a notion page",
    "notion page so you have a clean",
)

_META_REASONING_MARKERS = (
    "prompt injection attempt",
    "prompt-prompt injection",
    "i'm noticing this is a prompt",
    "i am noticing this is a prompt",
    "m noticing this is a prompt",
    "noticing this is a prompt",
    "trying to override my instructions",
    "override my instructions",
    "elaborate preamble is suspicious",
    "core request itself is legitimate",
    "actual task is straightforward",
    "the actual task is straightforward",
    "i should write the article",
    "ready to start writing",
    "ready to write",
    "i've got the key details",
    "ive got the key details",
    "looking past that framing",
    "but looking past that framing",
    "they're actually asking me",
    "they are actually asking me",
    "i'm ready to start writing",
    "i am ready to start writing",
    "i should respond",
    "i will respond",
    "i can answer",
    "let me write it directly",
    "write it directly in the conversation",
    "following many style guidelines",
    "a long article in",
    "style guidelines. let me",
    "in the conversation. let me",
)

_HEADING_START_RE = re.compile(r"#{1,6}\s+\S")

@dataclass(slots=True)
class NDJSONParseResult:
    text: str = ""
    thinking: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    notion_model: str | None = None
    line_count: int = 0
    event_type_counts: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _looks_like_search_preamble(fragment: str) -> bool:
    lower = fragment.strip().lower()
    if not lower or len(lower) > 600:
        return False
    return any(marker in lower for marker in _SEARCH_PREAMBLE_MARKERS)


def _looks_like_notion_page_preamble(fragment: str) -> bool:
    lower = fragment.strip().lower()
    if not lower or len(lower) > 600:
        return False
    return any(marker in lower for marker in _NOTION_PAGE_PREAMBLE_MARKERS)


def _starts_with_meta_reasoning(text: str) -> bool:
    """Return True if the beginning of the text reads like leaked model reasoning."""
    lower = text.strip().lower()
    if not lower:
        return False
    head = lower[:300]
    return any(marker in head for marker in _META_REASONING_MARKERS)


def _strip_meta_reasoning(text: str) -> str:
    """Strip Notion AI's internal reasoning that leaks as text before the real answer.

    The leaked reasoning is a contiguous block at the start of the output. The
    real answer usually starts at the first sentence boundary after the last
    reasoning marker (e.g. "...now." followed by the article text).
    """
    stripped = text.strip()
    if not stripped:
        return text
    if not _starts_with_meta_reasoning(stripped):
        return text

    # Prefer cutting at the first heading — the real article usually begins there.
    heading = _HEADING_START_RE.search(stripped)
    if heading and heading.start() > 0:
        candidate = stripped[heading.start() :].lstrip()
        if candidate:
            return candidate

    # Find the last occurrence of any meta-reasoning marker. We only search
    # in the first 1000 characters to prevent matching legitimate article content
    # (like "ready to write" or "respond") that appears later in a long text.
    head = stripped[:1000]
    lower_head = head.lower()
    last_marker_end = -1
    for marker in _META_REASONING_MARKERS:
        idx = lower_head.rfind(marker)
        if idx >= 0:
            last_marker_end = max(last_marker_end, idx + len(marker))

    # Search start: after the last marker, or from the beginning if no marker found.
    search_start = last_marker_end if last_marker_end >= 0 else 0
    tail = stripped[search_start:]

    # Find the first sentence boundary within the first 10 characters of tail.
    # The boundary is a period followed by whitespace and a capital letter, or
    # a period directly followed by an uppercase letter (e.g. "...conversation.Pernahkah...").
    # We restrict this to tail[:10] so we don't accidentally match legitimate
    # sentence boundaries further down in the article.
    match = re.search(r"\.\s*(?=[A-Z])", tail[:10])
    if match:
        start = search_start + match.end()
        candidate = stripped[start:].strip()
        if candidate and len(candidate) > 30:
            return candidate

    # Fallback: if we found markers but no clean boundary, drop everything
    # up to the end of the last marker.
    if last_marker_end >= 0:
        return stripped[last_marker_end:].strip()

    return ""



def clean_notion_output_text(text: str) -> str:
    """Remove Notion web-search, Notion-page, and meta-reasoning lead-ins that leak into assistant text."""
    if not text:
        return text

    stripped = text.strip()
    if not stripped:
        return text

    # If the model is leaking internal reasoning as text, suppress everything
    # until we can find the real start of the answer. While streaming, this
    # holds back output until the reasoning block ends.
    if _starts_with_meta_reasoning(stripped):
        cleaned_reasoning = _strip_meta_reasoning(stripped)
        if cleaned_reasoning:
            stripped = cleaned_reasoning
        else:
            return ""

    heading = _HEADING_START_RE.search(stripped)
    if heading and heading.start() > 0:
        before = stripped[: heading.start()].strip().rstrip(".")
        if _looks_like_search_preamble(before) or _looks_like_notion_page_preamble(before):
            return stripped[heading.start() :].lstrip()

    if "\n" in stripped:
        first_line, rest = stripped.split("\n", 1)
        if (
            (_looks_like_search_preamble(first_line) or _looks_like_notion_page_preamble(first_line))
            and rest.strip()
        ):
            return rest.strip()

    if (
        _looks_like_search_preamble(stripped) or _looks_like_notion_page_preamble(stripped)
    ) and not _HEADING_START_RE.search(stripped):
        return ""

    return stripped

def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class NDJSONStreamParser:
    def __init__(self) -> None:
        self._stored_text = ""
        self._stored_thinking = ""
        self._block_contents: dict[str, str] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.notion_model: str | None = None
        self.line_count = 0
        self.event_type_counts: dict[str, int] = {}
        self.tool_calls: list[dict[str, Any]] = []
        self._value_types: dict[str, str] = {}
        self._value_counts: dict[str, int] = {}
        self._section_count = 0
        self._tool_use_state: dict[str, dict[str, Any]] = {}

    @property
    def text(self) -> str:
        if self._block_contents:
            parts = []
            for s_idx in range(self._section_count):
                prefix = f"/s/{s_idx}"
                count = self._value_counts.get(prefix, 0)
                for v_idx in range(count):
                    path = f"{prefix}/value/{v_idx}"
                    if self._value_types.get(path) == "text":
                        parts.append(self._block_contents.get(path, ""))
            return "".join(parts)
        return self._stored_text

    @text.setter
    def text(self, value: str) -> None:
        self._stored_text = value

    @property
    def thinking(self) -> str:
        if self._block_contents:
            parts = []
            for s_idx in range(self._section_count):
                prefix = f"/s/{s_idx}"
                count = self._value_counts.get(prefix, 0)
                for v_idx in range(count):
                    path = f"{prefix}/value/{v_idx}"
                    if self._value_types.get(path) == "thinking":
                        parts.append(self._block_contents.get(path, ""))
            return "".join(parts)
        return self._stored_thinking

    @thinking.setter
    def thinking(self, value: str) -> None:
        self._stored_thinking = value

    def feed_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        self.line_count += 1
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return

        event_type = event.get("type")
        if not isinstance(event_type, str):
            return
        self.event_type_counts[event_type] = self.event_type_counts.get(event_type, 0) + 1

        if event_type == "error":
            msg = event.get("message") or event.get("data") or "unknown notion error"
            raise NotionChatError(f"Notion error: {msg}", status_code=502)
        if event_type == "premium-feature-unavailable":
            raise NotionChatError("Notion premium feature unavailable", status_code=402)

        if event_type == "patch":
            self._handle_patch(event)
        elif event_type in ("patch-start", "patch-sync"):
            self._handle_patch_start(event)
        elif event_type == "agent-inference":
            self._handle_agent_inference(event)
        elif event_type == "record-map":
            self._handle_record_map(event)

    def feed(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.feed_line(line)

    def finalize(self) -> NDJSONParseResult:
        return NDJSONParseResult(
            text=clean_notion_output_text(self.text),
            thinking=self.thinking,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            notion_model=self.notion_model,
            line_count=self.line_count,
            event_type_counts=dict(self.event_type_counts),
            tool_calls=list(self.tool_calls),
        )
    def _tool_prefix(self, path: str) -> str | None:
        if "/value/" not in path:
            return None
        return path[: path.index("/value/")]

    def _register_tool_use(self, prefix: str, entry: dict[str, Any]) -> None:
        name = entry.get("name")
        tool_id = entry.get("id")
        inp = entry.get("input")
        state = self._tool_use_state.setdefault(prefix, {})
        if isinstance(name, str):
            state["name"] = name
        if isinstance(tool_id, str):
            state["id"] = tool_id
        if inp is not None:
            state["input"] = inp
        if isinstance(state.get("name"), str):
            self._commit_tool_use(prefix)

    def _commit_tool_use(self, prefix: str) -> None:
        state = self._tool_use_state.get(prefix)
        if not state or not isinstance(state.get("name"), str):
            return
        inp = state.get("input")
        if isinstance(inp, dict):
            args = json.dumps(inp, ensure_ascii=False)
        elif isinstance(inp, str):
            args = inp
        else:
            args = "{}"
        call = {
            "id": state.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": state["name"], "arguments": args},
        }
        existing_ids = {c.get("id") for c in self.tool_calls}
        if call["id"] not in existing_ids:
            self.tool_calls.append(call)

    def _raise_premium_unavailable(self, entry: dict[str, Any]) -> None:
        avail = entry.get("featureAvailability") or {}
        limit = avail.get("limit") or {}
        current = limit.get("current")
        total = limit.get("total")
        upsell = avail.get("upsell") or {}
        product = upsell.get("product", "a paid plan")
        raise NotionChatError(
            f"Notion AI credits exhausted ({current}/{total} used). "
            f"Upgrade to Notion {product} or wait for your quota to reset.",
            status_code=402,
        )

    def _raise_inference_error(self, entry: dict[str, Any]) -> None:
        sub_type = entry.get("subType") or ""
        message = entry.get("message") or "Notion rejected the inference request"
        if entry.get("type") == "premium-feature-unavailable" or sub_type == "premium-feature-unavailable":
            self._raise_premium_unavailable(entry)
        if sub_type == "trust-rule-denied":
            raise NotionChatError(
                f"{message} Paste a fresh cookie from app.notion.com after opening Notion AI in the browser.",
                status_code=403,
            )
        raise NotionChatError(
            f"Notion error ({sub_type or 'unknown'}): {message}",
            status_code=502,
        )

    def _handle_patch_start(self, event: dict[str, Any]) -> None:
        data = event.get("data") or {}
        s = data.get("s")
        if isinstance(s, list):
            for entry in s:
                if isinstance(entry, dict):
                    if entry.get("type") == "premium-feature-unavailable":
                        self._raise_premium_unavailable(entry)
                    if entry.get("type") == "error":
                        self._raise_inference_error(entry)
            self._section_count = len(s)
            for i, sec in enumerate(s):
                self._value_counts.setdefault(f"/s/{i}", 0)
                if isinstance(sec, dict):
                    self._absorb_inline_section(i, sec)

    def _handle_patch(self, event: dict[str, Any]) -> None:
        ops = event.get("v")
        if not isinstance(ops, list):
            return
        for op in ops:
            if isinstance(op, dict):
                self._handle_patch_op(op)

    def _handle_patch_op(self, op: dict[str, Any]) -> None:
        o = op.get("o")
        p = op.get("p")
        v = op.get("v")
        if not isinstance(o, str) or not isinstance(p, str):
            return

        if o == "a" and p == "/s/-" and isinstance(v, dict):
            section_idx = self._section_count
            self._section_count += 1
            self._absorb_inline_section(section_idx, v)
            return

        if o in ("a", "p") and "/value/" in p and isinstance(v, dict):
            state_prefix = p[: p.index("/value/")]
            entry_type = v.get("type")
            
            # Extract the index or '-' from the path (e.g. /s/0/value/12 or /s/0/value/-)
            val_part = p[p.index("/value/") + 7 :]
            if "/" not in val_part:
                if val_part == "-":
                    idx = self._value_counts.get(state_prefix, 0)
                else:
                    try:
                        idx = int(val_part)
                    except ValueError:
                        idx = self._value_counts.get(state_prefix, 0)
                
                entry_path = f"{state_prefix}/value/{idx}"
                if isinstance(entry_type, str):
                    self._value_types[entry_path] = entry_type
                
                self._value_counts[state_prefix] = max(self._value_counts.get(state_prefix, 0), idx + 1)
                
                content = v.get("content")
                if entry_type == "tool_use":
                    self._register_tool_use(entry_path, v)
                    return
                if entry_type in ("text", "thinking"):
                    self._block_contents[entry_path] = content or ""
                return

        if o in ("a", "x", "p") and p.endswith("/name") and isinstance(v, str):
            prefix = self._tool_prefix(p)
            if prefix:
                self._tool_use_state.setdefault(prefix, {})["name"] = v
                self._commit_tool_use(prefix)
            return
        if o in ("a", "x", "p") and p.endswith("/input"):
            prefix = self._tool_prefix(p)
            if prefix:
                self._tool_use_state.setdefault(prefix, {})["input"] = v
                self._commit_tool_use(prefix)
            return
        if o in ("a", "x", "p") and p.endswith("/id") and isinstance(v, str):
            prefix = self._tool_prefix(p)
            if prefix:
                self._tool_use_state.setdefault(prefix, {})["id"] = v
            return

        if o == "a" and p.endswith("/inputTokens") and _is_int(v):
            self.input_tokens += int(v)
            return
        if o == "a" and p.endswith("/outputTokens") and _is_int(v):
            self.output_tokens += int(v)
            return
        if o == "a" and p.endswith("/model") and isinstance(v, str):
            self.notion_model = v
            return

        if "content" not in p or not isinstance(v, str):
            return
        entry_type = self._classify_content_path(p)
        if entry_type == "tool_use":
            return

        # Find the block path (parent of /content)
        idx = p.rfind("/content")
        block_path = p[:idx] if idx >= 0 else p

        if entry_type == "thinking":
            if o == "x":
                self._block_contents[block_path] = self._block_contents.get(block_path, "") + v
            elif o == "p":
                self._block_contents[block_path] = v
            return
        if entry_type != "text":
            return
        if o == "x":
            self._block_contents[block_path] = self._block_contents.get(block_path, "") + v
        elif o == "p":
            self._block_contents[block_path] = v

    def _classify_content_path(self, path: str) -> str:
        idx = path.rfind("/content")
        if idx < 0:
            return "text"
        return self._value_types.get(path[:idx], "text")

    def _absorb_inline_section(self, section_idx: int, section: dict[str, Any]) -> None:
        section_type = section.get("type")
        values = section.get("value")
        if not isinstance(values, list) or section_type not in (
            "agent-inference",
            "agent-reply",
            "assistant-reply",
        ):
            return
        section_prefix = f"/s/{section_idx}"
        for i, entry in enumerate(values):
            if not isinstance(entry, dict):
                continue
            etype = entry.get("type")
            entry_path = f"{section_prefix}/value/{i}"
            if isinstance(etype, str):
                self._value_types[entry_path] = etype
            if etype == "tool_use":
                self._register_tool_use(entry_path, entry)
                continue
            content = entry.get("content") or ""
            if etype in ("text", "thinking"):
                self._block_contents[entry_path] = content
        self._value_counts[section_prefix] = len(values)

    def _extract_step_text(self, step: dict[str, Any]) -> str | None:
        step_type = step.get("type")
        if step_type == "premium-feature-unavailable":
            self._raise_premium_unavailable(step)
        if step_type == "error":
            self._raise_inference_error(step)
        if step_type != "agent-inference":
            return None
        values = step.get("value")
        if not isinstance(values, list):
            return None
        parts: list[str] = []
        for entry in values:
            if isinstance(entry, dict) and entry.get("type") == "text":
                content = entry.get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
        return "".join(parts) if parts else None

    def _handle_record_map(self, event: dict[str, Any]) -> None:
        record_map = event.get("recordMap") or {}
        for msg in (record_map.get("thread_message") or {}).values():
            value = (msg.get("value") or {}).get("value") or {}
            step = value.get("step") or {}
            if not isinstance(step, dict):
                continue
            text = self._extract_step_text(step)
            if text:
                self.text = clean_notion_output_text(text)

    def _handle_agent_inference(self, event: dict[str, Any]) -> None:
        values = event.get("value")
        if isinstance(values, list):
            text_parts: list[str] = []
            for entry in values:
                if not isinstance(entry, dict):
                    continue
                etype = entry.get("type")
                content = entry.get("content")
                if not isinstance(content, str) or not content:
                    continue
                if etype == "text":
                    text_parts.append(content)
                elif etype == "thinking":
                    self.thinking = content
            if text_parts:
                self.text = clean_notion_output_text("".join(text_parts))
        if _is_int(event.get("inputTokens")):
            self.input_tokens += int(event["inputTokens"])
        if _is_int(event.get("outputTokens")):
            self.output_tokens += int(event["outputTokens"])
        model = event.get("model")
        if isinstance(model, str):
            self.notion_model = model
