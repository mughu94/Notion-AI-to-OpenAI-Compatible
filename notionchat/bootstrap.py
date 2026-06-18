from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from curl_cffi import requests

from notionchat.account import NotionAccount, parse_browser_cookie, save_notion_account
from notionchat.exceptions import NotionChatError

BASE_URL = "https://app.notion.com/api/v3"
DEFAULT_CLIENT_VERSION = "23.13.20260616.2105"


@dataclass(slots=True, frozen=True)
class Workspace:
    space_id: str
    space_view_id: str
    space_name: str
    domain: str = ""


def _bootstrap_headers(token_v2: str, browser_id: str, user_id: str | None) -> dict[str, str]:
    parts = [f"notion_browser_id={browser_id}"]
    if user_id:
        parts.append(f"notion_user_id={user_id}")
        parts.append(f'notion_users=[%22{user_id}%22]')
    parts.extend(["notion_check_cookie_consent=false", f"token_v2={token_v2}"])
    cookie = "; ".join(parts)
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "notion-audit-log-platform": "web",
        "notion-client-version": DEFAULT_CLIENT_VERSION,
        "origin": "https://app.notion.com",
        "referer": "https://app.notion.com/",
        "cookie": cookie,
    }


def _extract_user(record_map: dict[str, Any], user_id: str) -> tuple[str, str]:
    users = record_map.get("notion_user") or {}
    entry = users.get(user_id) or {}
    value = (entry.get("value") or {}).get("value") or {}
    name_list = value.get("name") or []
    name = name_list[0][0] if name_list and name_list[0] else ""
    email = value.get("email", "")
    return name, email


def _extract_workspaces(record_map: dict[str, Any]) -> list[Workspace]:
    spaces: list[Workspace] = []
    space_map = record_map.get("space") or {}
    view_map = record_map.get("space_view") or {}
    for view_id, view_entry in view_map.items():
        view_val = (view_entry.get("value") or {}).get("value") or {}
        space_id = view_val.get("space_id") or view_val.get("parent_id")
        if not space_id:
            continue
        space_entry = space_map.get(space_id) or {}
        space_val = (space_entry.get("value") or {}).get("value") or {}
        name_list = space_val.get("name") or []
        name = name_list[0][0] if name_list and name_list[0] else space_id
        domain = space_val.get("domain", "") or ""
        spaces.append(
            Workspace(
                space_id=space_id,
                space_view_id=view_id,
                space_name=name,
                domain=domain,
            )
        )
    return spaces


def bootstrap_from_cookie_sync(
    cookie: str,
    *,
    space_name: str | None = None,
    account_path: str = "notion_account.json",
) -> NotionAccount:
    """Sync variant used during server startup when space_id is missing."""
    parsed = parse_browser_cookie(cookie)
    token = parsed.get("token_v2")
    if not token:
        raise NotionChatError("Cookie missing token_v2", status_code=400)

    user_id = parsed.get("notion_user_id") or None
    browser_id = parsed.get("notion_browser_id") or str(uuid.uuid4())
    device_id = parsed.get("device_id") or str(uuid.uuid4())

    headers = _bootstrap_headers(token, browser_id, user_id)
    resp = requests.post(
        f"{BASE_URL}/loadUserContent",
        json={"cursor": {"stack": []}, "limit": 100},
        headers=headers,
        impersonate="chrome",
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise NotionChatError(
            f"loadUserContent failed ({resp.status_code}): {resp.text[:300]!r}",
            status_code=502,
        )
    data = resp.json()

    return _account_from_load_user_content(
        cookie=cookie,
        data=data,
        token=token,
        user_id=user_id,
        browser_id=browser_id,
        device_id=device_id,
        space_name=space_name,
        account_path=account_path,
    )


def _account_from_load_user_content(
    *,
    cookie: str,
    data: dict[str, Any],
    token: str,
    user_id: str | None,
    browser_id: str,
    device_id: str,
    space_name: str | None,
    account_path: str,
) -> NotionAccount:
    record_map = data.get("recordMap") or {}
    if not user_id:
        for uid in record_map.get("notion_user") or {}:
            user_id = uid
            break
    if not user_id:
        raise NotionChatError("Could not determine notion_user_id", status_code=502)

    user_name, user_email = _extract_user(record_map, user_id)
    workspaces = _extract_workspaces(record_map)
    if not workspaces:
        raise NotionChatError("No workspaces found for this account", status_code=502)

    chosen = workspaces[0]
    if space_name:
        matches = [w for w in workspaces if w.space_name.lower() == space_name.lower()]
        if matches:
            chosen = matches[0]
        else:
            names = ", ".join(w.space_name for w in workspaces)
            raise NotionChatError(f"Workspace {space_name!r} not found. Available: {names}", status_code=400)
    elif len(workspaces) > 1:
        names = ", ".join(w.space_name for w in workspaces)
        raise NotionChatError(
            f"Multiple workspaces found. Set NOTION_SPACE_NAME or run init with --space-name. Available: {names}",
            status_code=400,
        )

    acc = NotionAccount(
        token_v2=token,
        full_cookie=cookie,
        user_id=user_id,
        user_name=user_name,
        user_email=user_email,
        space_id=chosen.space_id,
        space_name=chosen.space_name,
        space_view_id=chosen.space_view_id,
        browser_id=browser_id,
        device_id=device_id,
        client_version=DEFAULT_CLIENT_VERSION,
    )
    save_notion_account(acc, account_path)
    return acc


async def bootstrap_from_cookie(
    cookie: str,
    *,
    space_name: str | None = None,
    account_path: str = "notion_account.json",
) -> NotionAccount:
    return await asyncio.to_thread(
        bootstrap_from_cookie_sync,
        cookie,
        space_name=space_name,
        account_path=account_path,
    )
