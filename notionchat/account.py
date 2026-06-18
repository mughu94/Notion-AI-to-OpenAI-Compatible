from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from notionchat.exceptions import NotionChatError


@dataclass(slots=True, frozen=True)
class NotionAccount:
    token_v2: str
    full_cookie: str = ""
    user_id: str = ""
    user_name: str = ""
    user_email: str = ""
    space_id: str = ""
    space_name: str = ""
    space_view_id: str = ""
    browser_id: str = ""
    device_id: str = ""
    client_version: str = "23.13.20260616.2105"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    )
    timezone: str = "America/Los_Angeles"
    default_model: str = "ambrosia-tart-high"
    extras: dict[str, Any] = field(default_factory=dict)


def parse_browser_cookie(cookie: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in cookie.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, _, value = part.partition("=")
        out[name.strip()] = value.strip()
    return out


def load_notion_account(path: Path | str) -> NotionAccount:
    p = Path(path).expanduser()
    if not p.exists():
        raise NotionChatError(f"Account file not found: {p}", status_code=500)
    try:
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise NotionChatError(f"Invalid account JSON: {e}", status_code=500) from e

    for field_name in ("token_v2", "user_id", "space_id"):
        if not data.get(field_name):
            raise NotionChatError(
                f"Account file missing required field: {field_name}",
                status_code=500,
            )

    known = {f.name for f in NotionAccount.__dataclass_fields__.values()} - {"extras"}
    kwargs = {k: data[k] for k in known if k in data}
    extras = {k: v for k, v in data.items() if k not in known}
    return NotionAccount(**kwargs, extras=extras)


def save_notion_account(acc: NotionAccount, path: Path | str) -> None:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    for f in NotionAccount.__dataclass_fields__.values():
        if f.name == "extras":
            continue
        data[f.name] = getattr(acc, f.name)
    data.update(acc.extras)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_cookie_header(acc: NotionAccount) -> str:
    if acc.full_cookie:
        return acc.full_cookie
    parts = [
        f"notion_browser_id={acc.browser_id}",
        f"device_id={acc.device_id}",
        f"notion_user_id={acc.user_id}",
        f'notion_users=[%22{acc.user_id}%22]',
        "notion_check_cookie_consent=false",
        "notion_locale=en-US/autodetect",
        f"token_v2={acc.token_v2}",
    ]
    return "; ".join(parts)
