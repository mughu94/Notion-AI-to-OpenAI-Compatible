from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from notionchat.exceptions import NotionChatError

log = logging.getLogger(__name__)


@dataclass(slots=True)
class ThreadState:
    thread_id: str
    config_id: str
    context_id: str
    original_datetime: str
    notion_model: str
    updated_config_ids: list[str] = field(default_factory=list)
    last_activity_iso: str = ""


def thread_state_path(thread_id: str, base_dir: Path) -> Path:
    return base_dir / f"{thread_id}.json"


def save_thread_state(state: ThreadState, base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    p = thread_state_path(state.thread_id, base_dir)
    p.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def load_thread_state(thread_id: str, base_dir: Path) -> ThreadState:
    p = thread_state_path(thread_id, base_dir)
    if not p.exists():
        raise NotionChatError(
            f"No saved thread state for {thread_id}. Start a new conversation.",
            status_code=400,
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise NotionChatError(f"Corrupt thread state: {e}", status_code=500) from e
    return ThreadState(**data)
