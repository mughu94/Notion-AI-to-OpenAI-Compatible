from __future__ import annotations

import argparse
import asyncio
import sys

import uvicorn

from notionchat.bootstrap import bootstrap_from_cookie
from notionchat.config import load_settings
from notionchat.openai_api import create_app


def cmd_serve(_: argparse.Namespace) -> int:
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    cookie = args.cookie
    if cookie == "-":
        cookie = sys.stdin.read().strip()
    if not cookie:
        print("Error: provide --cookie or pipe cookie via stdin", file=sys.stderr)
        return 1

    async def run() -> None:
        acc = await bootstrap_from_cookie(
            cookie,
            space_name=args.space_name,
            account_path=args.account,
        )
        print(f"Saved account for workspace {acc.space_name!r} ({acc.space_id})")
        print(f"  user: {acc.user_name or acc.user_id}")
        print(f"  file: {args.account}")

    asyncio.run(run())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="notionchat", description="Notion AI OpenAI-compatible API")
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    serve_p.set_defaults(func=cmd_serve)

    init_p = sub.add_parser("init", help="Bootstrap notion_account.json from browser cookie")
    init_p.add_argument("--cookie", required=True, help='Full document.cookie string, or "-" for stdin')
    init_p.add_argument("--space-name", default=None, help="Workspace name when multiple exist")
    init_p.add_argument("--account", default="notion_account.json", help="Output account file path")
    init_p.set_defaults(func=cmd_init)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
