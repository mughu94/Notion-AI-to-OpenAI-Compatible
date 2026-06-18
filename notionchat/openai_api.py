from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

from notionchat.client import ChatResult, NotionAIClient
from notionchat.config import Settings, load_account_from_env, load_settings
from notionchat.exceptions import NotionChatError
from notionchat.models import (
    cache_openai_models,
    get_cached_alias_map,
    get_cached_openai_models,
    list_openai_models,
    list_openai_models_from_notion,
    normalize_request_model,
    parse_available_models,
    resolve_model,
)
from notionchat.tools import (
    bridge_ide_agent_response,
    build_tool_denial_retry_append,
    client_tool_names,
    is_ide_agent_messages,
    looks_like_coding_task_prompt,
    looks_like_tool_denial,
    normalize_tools,
    prepare_chat_input,
)

log = logging.getLogger(__name__)

_session_threads: dict[str, str] = {}
_session_models: dict[str, str] = {}


class FunctionDetails(BaseModel):
    name: str
    arguments: str = "{}"


class ToolCallPart(BaseModel):
    id: str
    type: str = "function"
    function: FunctionDetails


class ToolFunctionSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: str = "function"
    function: ToolFunctionSchema


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[ToolCallPart] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "notion-ai"
    messages: list[ChatMessage]
    stream: bool = False
    user: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = True


def _tools_payload(req: ChatCompletionRequest) -> list[dict[str, Any]]:
    if not req.tools:
        return []
    out: list[dict[str, Any]] = []
    for item in req.tools:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, ToolDefinition):
            out.append(item.model_dump())
        elif hasattr(item, "model_dump"):
            out.append(item.model_dump())
    return out


def _resolved_request_model(req: ChatCompletionRequest, settings: Settings) -> str:
    return resolve_model(
        normalize_request_model(req.model) or settings.default_model,
        default=settings.default_model,
        alias_map=get_cached_alias_map(),
    )


def _resolve_thread_id(req: ChatCompletionRequest, settings: Settings) -> str | None:
    if not req.user:
        return None

    resolved = _resolved_request_model(req, settings)
    previous = _session_models.get(req.user)
    if previous and previous != resolved:
        log.info(
            "Session %r model changed %r -> %r — dropping Notion thread",
            req.user,
            previous,
            resolved,
        )
        _session_threads.pop(req.user, None)
    _session_models[req.user] = resolved

    return _session_threads.get(req.user)


def _remember_thread(req: ChatCompletionRequest, thread_id: str, settings: Settings) -> None:
    if req.user:
        _session_threads[req.user] = thread_id
        _session_models[req.user] = _resolved_request_model(req, settings)


async def _ensure_model_aliases(client: NotionAIClient, settings: Settings) -> None:
    if get_cached_alias_map() is not None:
        return
    try:
        raw = await client.fetch_available_models()
        data = list_openai_models_from_notion(
            raw,
            default_notion_id=settings.default_model,
        )
        cache_openai_models(data, parse_available_models(raw))
    except NotionChatError as e:
        log.warning("Could not prefetch Notion model aliases: %s", e)


def _assistant_message(result: ChatResult) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant"}
    if result.tool_calls:
        msg["content"] = result.text
        msg["tool_calls"] = result.tool_calls
    else:
        msg["content"] = result.text or ""
    return msg


def _usage(result: ChatResult) -> dict[str, int]:
    return {
        "prompt_tokens": result.input_tokens,
        "completion_tokens": result.output_tokens,
        "total_tokens": result.input_tokens + result.output_tokens,
    }


def _chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _bridge_ide_agent(
    result: ChatResult,
    *,
    req: ChatCompletionRequest,
    tools: list[dict[str, Any]],
    prompt: str,
    ide_agent: bool,
    tools_active: bool,
) -> ChatResult:
    if not ide_agent or not tools_active:
        return result
    text, tool_calls = bridge_ide_agent_response(
        messages=req.messages,
        notion_text=result.text,
        notion_tool_calls=result.tool_calls,
        client_tools=tools,
        prompt=prompt,
    )
    if tool_calls:
        log.info(
            "IDE bridge tool_calls=%s (notion text_len=%s)",
            [(tc.get("function") or {}).get("name") for tc in tool_calls],
            len(result.text or ""),
        )
        return ChatResult(
            text=text,
            thread_id=result.thread_id,
            model=result.model,
            tool_calls=tool_calls,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    if looks_like_tool_denial(result.text):
        log.info("IDE bridge suppressed Notion tool denial (text_len=%s)", len(result.text or ""))
        return ChatResult(
            text=None,
            thread_id=result.thread_id,
            model=result.model,
            tool_calls=None,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    return ChatResult(
        text=text if text is not None else result.text,
        thread_id=result.thread_id,
        model=result.model,
        tool_calls=None,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


async def _bridge_ide_agent_async(
    client: NotionAIClient,
    result: ChatResult,
    *,
    req: ChatCompletionRequest,
    tools: list[dict[str, Any]],
    prompt: str,
    system: str | None,
    ide_agent: bool,
    tools_active: bool,
) -> ChatResult:
    bridged = _bridge_ide_agent(
        result,
        req=req,
        tools=tools,
        prompt=prompt,
        ide_agent=ide_agent,
        tools_active=tools_active,
    )
    if bridged.tool_calls or not ide_agent or not tools_active:
        return bridged

    should_retry = looks_like_tool_denial(result.text) or looks_like_coding_task_prompt(prompt)
    if not should_retry:
        return bridged

    log.info("IDE bridge retry: no tool_calls after compile (denial=%s)", looks_like_tool_denial(result.text))
    retry_system = (system or "").strip()
    append = build_tool_denial_retry_append()
    retry_system = f"{retry_system}\n\n{append}".strip() if retry_system else append
    retry = await client.complete(
        prompt=prompt,
        system=retry_system,
        model=req.model,
        thread_id=None,
        tools_active=tools_active,
        ide_agent_mode=ide_agent,
        client_tools=tools,
    )
    return _bridge_ide_agent(
        retry,
        req=req,
        tools=tools,
        prompt=prompt,
        ide_agent=ide_agent,
        tools_active=tools_active,
    )


async def _stream_openai(
    client: NotionAIClient,
    req: ChatCompletionRequest,
    system: str | None,
    prompt: str,
    thread_id: str | None,
    settings: Settings,
    *,
    tools: list[dict[str, Any]],
    tools_active: bool,
    ide_agent: bool,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        deltas, active_thread_id, finalize = await client.stream_deltas(
            prompt=prompt,
            system=system,
            model=req.model,
            thread_id=thread_id,
            tools_active=tools_active,
            ide_agent_mode=ide_agent,
            client_tools=tools,
        )

        if tools_active:
            buffered: list[str] = []
            async for piece in deltas:
                buffered.append(piece)
            try:
                result = finalize()
            except NotionChatError:
                empty = ChatResult(
                    text=None,
                    thread_id=active_thread_id,
                    model=req.model,
                    tool_calls=None,
                )
                result = await _bridge_ide_agent_async(
                    client,
                    empty,
                    req=req,
                    tools=tools,
                    prompt=prompt,
                    system=system,
                    ide_agent=ide_agent,
                    tools_active=tools_active,
                )
                if not result.tool_calls and not (result.text or "").strip():
                    raise
            else:
                result = await _bridge_ide_agent_async(
                    client,
                    result,
                    req=req,
                    tools=tools,
                    prompt=prompt,
                    system=system,
                    ide_agent=ide_agent,
                    tools_active=tools_active,
                )
            if not ide_agent and not tools_active:
                _remember_thread(req, result.thread_id, settings)

            if result.tool_calls:
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={"role": "assistant", "content": None},
                )
                for index, tc in enumerate(result.tool_calls):
                    fn = tc.get("function") or {}
                    yield _chunk(
                        completion_id=completion_id,
                        created=created,
                        model=req.model,
                        delta={
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": tc.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": fn.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                    )
                    args = str(fn.get("arguments", ""))
                    step = max(1, len(args) // 4)
                    for pos in range(0, len(args), step):
                        yield _chunk(
                            completion_id=completion_id,
                            created=created,
                            model=req.model,
                            delta={
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "function": {"arguments": args[pos : pos + step]},
                                    }
                                ]
                            },
                        )
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={},
                    finish_reason="tool_calls",
                )
            else:
                pieces = buffered
                if not pieces and result.text:
                    pieces = [result.text]
                for piece in pieces:
                    yield _chunk(
                        completion_id=completion_id,
                        created=created,
                        model=req.model,
                        delta={"content": piece},
                    )
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={},
                    finish_reason="stop",
                )
        else:
            async for piece in deltas:
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={"content": piece},
                )
            result = finalize()
            if not ide_agent and not tools_active:
                _remember_thread(req, result.thread_id, settings)
            finish_reason = "tool_calls" if result.tool_calls else "stop"
            if result.tool_calls:
                yield _chunk(
                    completion_id=completion_id,
                    created=created,
                    model=req.model,
                    delta={"role": "assistant", "tool_calls": result.tool_calls},
                )
            yield _chunk(
                completion_id=completion_id,
                created=created,
                model=req.model,
                delta={},
                finish_reason=finish_reason,
            )

        yield "data: [DONE]\n\n"
    except NotionChatError as e:
        err = {"error": {"message": str(e), "type": "notion_error", "code": e.status_code}}
        yield f"data: {json.dumps(err)}\n\n"
    finally:
        await client.aclose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="NotionChat", version="0.1.0")

    def verify_key(authorization: str | None = Header(default=None)) -> None:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing Bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        if token != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def get_client() -> NotionAIClient:
        account = load_account_from_env(settings)
        settings.thread_state_dir.mkdir(parents=True, exist_ok=True)
        return NotionAIClient(
            account,
            base_url=settings.base_url,
            thread_state_dir=settings.thread_state_dir,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(_: None = Depends(verify_key)) -> dict[str, Any]:
        cached = get_cached_openai_models()
        if cached is not None:
            return {"object": "list", "data": cached}

        client = get_client()
        try:
            raw = await client.fetch_available_models()
            data = list_openai_models_from_notion(
                raw,
                default_notion_id=settings.default_model,
            )
            cache_openai_models(data, parse_available_models(raw))
            return {"object": "list", "data": data}
        except NotionChatError as e:
            log.warning("getAvailableModels failed (%s), using static model list", e)
            return {
                "object": "list",
                "data": list_openai_models(settings.default_model),
            }
        finally:
            await client.aclose()

    @app.post("/v1/chat/completions")
    async def chat_completions(
        req: ChatCompletionRequest,
        _: None = Depends(verify_key),
    ) -> Any:
        try:
            tools = normalize_tools(_tools_payload(req))
            system, prompt, tools_active, ide_agent, tools = prepare_chat_input(
                req.messages,
                tools=tools,
                tool_choice=req.tool_choice,
            )
            if not tools and is_ide_agent_messages(req.messages):
                log.warning("Cursor-like request but tools[] empty after prepare_chat_input")
            log.info(
                "chat stream=%s model=%s resolved=%s tools=%d tools_active=%s ide_agent=%s tool_names=%s msgs=%d",
                req.stream,
                normalize_request_model(req.model) or req.model,
                _resolved_request_model(req, settings),
                len(tools),
                tools_active,
                ide_agent,
                sorted(client_tool_names(tools))[:8],
                len(req.messages),
            )
            thread_id = None if (ide_agent or tools_active) else _resolve_thread_id(req, settings)
            client = get_client()
            try:
                await _ensure_model_aliases(client, settings)
                if req.stream:
                    return StreamingResponse(
                        _stream_openai(
                            client,
                            req,
                            system,
                            prompt,
                            thread_id,
                            settings,
                            tools=tools,
                            tools_active=tools_active,
                            ide_agent=ide_agent,
                        ),
                        media_type="text/event-stream",
                    )
                result = await client.complete(
                    prompt=prompt,
                    system=system,
                    model=req.model,
                    thread_id=thread_id,
                    tools_active=tools_active,
                    ide_agent_mode=ide_agent,
                    client_tools=tools,
                )
                result = await _bridge_ide_agent_async(
                    client,
                    result,
                    req=req,
                    tools=tools,
                    prompt=prompt,
                    system=system,
                    ide_agent=ide_agent,
                    tools_active=tools_active,
                )
                log.info(
                    "chat result text_len=%s tool_calls=%s",
                    len(result.text or ""),
                    [((tc.get("function") or {}).get("name")) for tc in (result.tool_calls or [])],
                )
                if not ide_agent and not tools_active:
                    _remember_thread(req, result.thread_id, settings)
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                finish_reason = "tool_calls" if result.tool_calls else "stop"
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": _assistant_message(result),
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": _usage(result),
                }
            finally:
                await client.aclose()
        except NotionChatError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e

    return app
