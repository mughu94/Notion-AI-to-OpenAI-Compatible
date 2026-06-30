from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import uvicorn

from notionchat.bootstrap import bootstrap_from_cookie
from notionchat.config import load_settings
from notionchat.openai_api import create_app
from notionchat.setup_cli import run_interactive_setup


def cmd_serve(_: argparse.Namespace) -> int:
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    return asyncio.run(
        run_interactive_setup(
            env_path=Path(args.env) if args.env else None,
            account_path=Path(args.account) if args.account else None,
            cookie=args.cookie,
            space_name=args.space_name,
            api_key=args.api_key,
            host=args.host,
            port=args.port,
            write_cookie_to_env=args.write_cookie if args.write_cookie is not None else None,
            force=args.force,
            yes=args.yes,
        )
    )


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

    setup_p = sub.add_parser("setup", help="Interactive wizard: cookie → account file → .env")
    setup_p.add_argument("--env", default=".env", help="Path to write environment file (default: .env)")
    setup_p.add_argument("--account", default="notion_account.json", help="Output account file path")
    setup_p.add_argument("--cookie", default=None, help="Skip cookie prompt and use this value")
    setup_p.add_argument("--space-name", default=None, help="Workspace name when multiple exist")
    setup_p.add_argument("--api-key", default=None, help="API key for NOTIONCHAT_API_KEY")
    setup_p.add_argument("--host", default=None, help="Bind host for NOTIONCHAT_HOST")
    setup_p.add_argument("--port", default=None, help="Port for NOTIONCHAT_PORT")
    setup_p.add_argument(
        "--write-cookie",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store NOTION_COOKIE in .env (default: ask interactively)",
    )
    setup_p.add_argument("--force", action="store_true", help="Overwrite .env without asking")
    setup_p.add_argument("-y", "--yes", action="store_true", help="Accept defaults with minimal prompts")
    setup_p.set_defaults(func=cmd_setup)

    init_p = sub.add_parser("init", help="Bootstrap notion_account.json from browser cookie")
    init_p.add_argument("--cookie", required=True, help='Full document.cookie string, or "-" for stdin')
    init_p.add_argument("--space-name", default=None, help="Workspace name when multiple exist")
    init_p.add_argument("--account", default="notion_account.json", help="Output account file path")
    init_p.set_defaults(func=cmd_init)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        _hint = (
            "\nNew here? Run the interactive setup wizard:\n"
            "  python -m notionchat setup\n"
        )
        print(_hint, file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
