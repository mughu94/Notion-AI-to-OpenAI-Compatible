from __future__ import annotations

from typing import Any

from curl_cffi.requests import AsyncSession, Response

BROWSER_IMPERSONATE = "chrome"
DEFAULT_TIMEOUT = 300.0


class NotionHttpStatusError(Exception):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:200]!r}")


class NotionHttpClient:
    """Impersonates Chrome TLS — required for Notion AI on Business plans."""

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout
        self._session: AsyncSession | None = None

    async def _session_or_create(self) -> AsyncSession:
        if self._session is None:
            self._session = AsyncSession(timeout=self._timeout)
        return self._session

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def post_json(self, url: str, *, json: dict[str, Any], headers: dict[str, str]) -> Any:
        session = await self._session_or_create()
        resp = await session.post(
            url,
            json=json,
            headers=headers,
            impersonate=BROWSER_IMPERSONATE,
        )
        if resp.status_code != 200:
            raise NotionHttpStatusError(resp.status_code, resp.text or "")
        return resp.json()

    async def post_stream(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> Response:
        session = await self._session_or_create()
        return await session.post(
            url,
            json=json,
            headers=headers,
            impersonate=BROWSER_IMPERSONATE,
            stream=True,
        )
