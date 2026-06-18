from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from notionchat.account import NotionAccount, build_cookie_header
from notionchat.exceptions import NotionChatError
from notionchat.models import get_cached_alias_map, normalize_request_model, resolve_model
from notionchat.ndjson import NDJSONStreamParser, clean_notion_output_text
from notionchat.notion_http import NotionHttpClient, NotionHttpStatusError
from notionchat.thread_state import ThreadState, load_thread_state, save_thread_state
from notionchat.tools import merge_tool_calls
from notionchat.transcript import (
    _now_iso,
    build_full_transcript,
    build_inference_request,
    build_partial_transcript,
    new_uuid,
)

log = logging.getLogger(__name__)


def _empty_response_message(result, thread_id: str) -> str:
    if result.tool_calls:
        return ""
    if not result.line_count:
        return (
            "Notion returned no stream data. Check that space_id is set "
            "(run `python -m notionchat init --cookie ...`) and your cookie is fresh."
        )
    events = ", ".join(f"{k}={v}" for k, v in sorted(result.event_type_counts.items()))
    return (
        f"Notion returned empty assistant text (thread={thread_id}, events: {events or 'none'}). "
        "Your AI credits may be exhausted, or the response format changed."
    )


@dataclass(slots=True)
class ChatResult:
    text: str | None
    thread_id: str
    model: str
    tool_calls: list[dict[str, Any]] | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def build_headers(acc: NotionAccount, *, accept: str = "application/x-ndjson") -> dict[str, str]:
    return {
        "accept": accept,
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/json",
        "notion-audit-log-platform": "web",
        "notion-client-version": acc.client_version,
        "origin": "https://app.notion.com",
        "referer": "https://app.notion.com/ai",
        "user-agent": acc.user_agent,
        "x-notion-active-user-header": acc.user_id,
        "x-notion-space-id": acc.space_id,
        "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "cookie": build_cookie_header(acc),
    }


class NotionAIClient:
    def __init__(
        self,
        account: NotionAccount,
        *,
        base_url: str,
        thread_state_dir: Path,
        http_client: NotionHttpClient | None = None,
    ):
        self.account = account
        self.base_url = base_url.rstrip("/")
        self.thread_state_dir = thread_state_dir
        self._client = http_client
        self._owns_client = http_client is None

    def _get_client(self) -> NotionHttpClient:
        if self._client is None:
            self._client = NotionHttpClient()
        return self._client

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _prepare(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str | None,
        thread_id: str | None,
        ide_agent_mode: bool = False,
    ) -> tuple[dict[str, Any], dict[str, str], str, str, Callable[[], None]]:
        acc = self.account
        joined = f"{system}\n\n{prompt}" if system else prompt
        if not joined.strip():
            raise NotionChatError("Empty prompt", status_code=400)

        notion_model = resolve_model(
            normalize_request_model(model) or acc.default_model,
            default=acc.default_model,
            alias_map=get_cached_alias_map(),
        )
        log.info("Notion model: request=%r -> %r", model, notion_model)

        reuse_thread_id = thread_id
        prior: ThreadState | None = None
        if thread_id:
            prior = load_thread_state(thread_id, self.thread_state_dir)
            if prior.notion_model != notion_model:
                log.info(
                    "Model changed on thread %s (%r -> %r) — starting new Notion thread",
                    thread_id,
                    prior.notion_model,
                    notion_model,
                )
                reuse_thread_id = None

        if reuse_thread_id and prior:
            updated_ids = [*prior.updated_config_ids, new_uuid()]
            transcript = build_partial_transcript(
                acc,
                new_user_text=joined,
                notion_model=notion_model,
                config_id=prior.config_id,
                context_id=prior.context_id,
                updated_config_ids=updated_ids,
                original_datetime=prior.original_datetime,
                ide_agent_mode=ide_agent_mode,
            )
            active_thread_id = reuse_thread_id
            create_thread = False
            is_partial = True

            def save_state() -> None:
                prior.updated_config_ids = updated_ids
                prior.notion_model = notion_model
                prior.last_activity_iso = _now_iso(acc.timezone)
                save_thread_state(prior, self.thread_state_dir)
        else:
            config_id = new_uuid()
            context_id = new_uuid()
            first_dt = _now_iso(acc.timezone)
            transcript = build_full_transcript(
                acc,
                user_text=joined,
                notion_model=notion_model,
                config_id=config_id,
                context_id=context_id,
                now=first_dt,
                ide_agent_mode=ide_agent_mode,
            )
            active_thread_id = new_uuid()
            create_thread = True
            is_partial = False

            def save_state() -> None:
                save_thread_state(
                    ThreadState(
                        thread_id=active_thread_id,
                        config_id=config_id,
                        context_id=context_id,
                        original_datetime=first_dt,
                        notion_model=notion_model,
                    ),
                    self.thread_state_dir,
                )

        body = build_inference_request(
            acc,
            transcript=transcript,
            thread_id=active_thread_id,
            create_thread=create_thread,
            is_partial_transcript=is_partial,
        )
        headers = build_headers(acc)
        return body, headers, active_thread_id, notion_model, save_state

    def _raise_http(self, status_code: int, body: str) -> None:
        snippet = body[:500]
        if status_code in (401, 403):
            raise NotionChatError(
                f"Notion auth failed ({status_code}). Refresh token_v2 cookie. {snippet!r}",
                status_code=401,
            )
        raise NotionChatError(f"Notion API {status_code}: {snippet!r}", status_code=502)

    async def _consume_stream(
        self,
        resp,
        parser: NDJSONStreamParser,
        *,
        on_delta: Callable[[str], None] | None = None,
    ) -> None:
        last_len = 0
        async for line in resp.aiter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            parser.feed_line(line)
            if on_delta and len(parser.text) > last_len:
                on_delta(parser.text[last_len:])
                last_len = len(parser.text)

    async def _run_inference(
        self,
        *,
        prompt: str,
        system: str | None,
        model: str | None,
        thread_id: str | None,
        ide_agent_mode: bool,
        on_delta: Callable[[str], None] | None = None,
        tools_active: bool,
        client_tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        body, headers, active_thread_id, notion_model, save_state = self._prepare(
            prompt=prompt,
            system=system,
            model=model,
            thread_id=thread_id,
            ide_agent_mode=ide_agent_mode,
        )
        url = f"{self.base_url}/runInferenceTranscript"
        parser = NDJSONStreamParser()
        client = self._get_client()
        try:
            resp = await client.post_stream(url, json=body, headers=headers)
            if resp.status_code != 200:
                self._raise_http(resp.status_code, await resp.atext())
            await self._consume_stream(resp, parser, on_delta=on_delta)
        except NotionHttpStatusError as e:
            self._raise_http(e.status_code, e.body)
        except NotionChatError:
            raise
        except Exception as e:
            raise NotionChatError(f"Notion transport error: {e}", status_code=502) from e

        result = parser.finalize()
        raw_text = result.text
        content, tool_calls = merge_tool_calls(
            text=raw_text,
            ndjson_tool_calls=result.tool_calls,
            tools_active=tools_active,
            client_tools=client_tools,
            prompt=prompt,
            ide_agent=ide_agent_mode,
        )
        if not content and not tool_calls:
            raise NotionChatError(_empty_response_message(result, active_thread_id), status_code=502)
        save_state()
        return ChatResult(
            text=raw_text if ide_agent_mode else content,
            thread_id=active_thread_id,
            model=result.notion_model or notion_model,
            tool_calls=tool_calls or None,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        thread_id: str | None = None,
        on_delta: Callable[[str], None] | None = None,
        tools_active: bool = False,
        ide_agent_mode: bool = False,
        client_tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        result = await self._run_inference(
            prompt=prompt,
            system=system,
            model=model,
            thread_id=thread_id,
            ide_agent_mode=ide_agent_mode,
            on_delta=on_delta,
            tools_active=tools_active,
            client_tools=client_tools,
        )
        return result

    async def stream_deltas(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        thread_id: str | None = None,
        tools_active: bool = False,
        ide_agent_mode: bool = False,
        client_tools: list[dict[str, Any]] | None = None,
    ) -> tuple[AsyncIterator[str], str, Callable[[], ChatResult]]:
        body, headers, active_thread_id, notion_model, save_state = self._prepare(
            prompt=prompt,
            system=system,
            model=model,
            thread_id=thread_id,
            ide_agent_mode=ide_agent_mode,
        )
        url = f"{self.base_url}/runInferenceTranscript"
        client = self._get_client()
        parser = NDJSONStreamParser()
        last_emitted_clean = ""
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        http_error: list[BaseException] = []

        async def producer() -> None:
            nonlocal last_emitted_clean
            try:
                resp = await client.post_stream(url, json=body, headers=headers)
                if resp.status_code != 200:
                    self._raise_http(resp.status_code, await resp.atext())
                async for line in resp.aiter_lines():
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    parser.feed_line(line)
                    cleaned = clean_notion_output_text(parser.text)
                    if not cleaned:
                        continue
                    if cleaned.startswith(last_emitted_clean):
                        delta = cleaned[len(last_emitted_clean) :]
                    else:
                        delta = cleaned
                    if delta:
                        await queue.put(delta)
                        last_emitted_clean = cleaned
            except BaseException as e:
                http_error.append(e)
            finally:
                await queue.put(None)

        async def consumer() -> AsyncIterator[str]:
            task = asyncio.create_task(producer())
            try:
                while True:
                    chunk = await queue.get()
                    if chunk is None:
                        break
                    yield chunk
                if http_error:
                    raise http_error[0]
            finally:
                await task

        def finalize_result() -> ChatResult:
            result = parser.finalize()
            raw_text = result.text
            content, tool_calls = merge_tool_calls(
                text=raw_text,
                ndjson_tool_calls=result.tool_calls,
                tools_active=tools_active,
                client_tools=client_tools,
                prompt=prompt,
                ide_agent=ide_agent_mode,
            )
            if not content and not tool_calls:
                raise NotionChatError(
                    _empty_response_message(result, active_thread_id),
                    status_code=502,
                )
            save_state()
            return ChatResult(
                text=raw_text if ide_agent_mode else content,
                thread_id=active_thread_id,
                model=result.notion_model or notion_model,
                tool_calls=tool_calls or None,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        return consumer(), active_thread_id, finalize_result

    async def fetch_available_models(self) -> dict[str, Any]:
        url = f"{self.base_url}/getAvailableModels"
        headers = build_headers(self.account, accept="application/json")
        client = self._get_client()
        try:
            return await client.post_json(
                url,
                json={"spaceId": self.account.space_id},
                headers=headers,
            )
        except NotionHttpStatusError as e:
            raise NotionChatError(f"getAvailableModels failed: {e.status_code}", status_code=502) from e
