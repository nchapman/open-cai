"""Small OpenRouter chat-completions client."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "deepseek/deepseek-v3.2"
COMPILER_MODEL = "deepseek/deepseek-v4-pro"
GUIDE_APPLICATION_MODEL = "deepseek/deepseek-v3.2"
COMPILER_REASONING = {"effort": "high", "exclude": True}
GUIDE_APPLICATION_REASONING = {"effort": "none", "exclude": True}


Message = Mapping[str, str]


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter inference fails."""


@dataclasses.dataclass(frozen=True)
class OpenRouterSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    app_url: str | None = None
    app_title: str | None = None


def load_env_file(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs into the process environment."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def settings_from_env(
    env_path: str | Path = ".env",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> OpenRouterSettings:
    """Build OpenRouter settings from environment variables and `.env`."""

    load_env_file(env_path)
    explicit_base_url = base_url is not None
    resolved_base_url = (base_url or os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL)).strip().rstrip("/")
    if api_key is not None:
        resolved_api_key = api_key.strip()
    elif explicit_base_url and resolved_base_url != DEFAULT_BASE_URL:
        resolved_api_key = "local-token"
    else:
        resolved_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not resolved_api_key and resolved_base_url == DEFAULT_BASE_URL:
        raise OpenRouterError("OPENROUTER_API_KEY is not set")
    if not resolved_api_key:
        resolved_api_key = "local-token"

    return OpenRouterSettings(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        app_url=os.getenv("APP_URL"),
        app_title=os.getenv("APP_TITLE"),
    )


class OpenRouterClient:
    """Minimal OpenAI-compatible chat client for OpenRouter."""

    def __init__(self, settings: OpenRouterSettings) -> None:
        self.settings = settings

    def chat(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        response_format: Mapping[str, object] | None = None,
        reasoning: Mapping[str, object] | None = None,
    ) -> str:
        payload = {
            "model": model or DEFAULT_CHAT_MODEL,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if reasoning is not None:
            payload["reasoning"] = dict(reasoning)
        response = self._post("/chat/completions", payload)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"unexpected OpenRouter response: {response!r}") from exc

    def _post(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        url = f"{self.settings.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.app_url:
            headers["HTTP-Referer"] = self.settings.app_url
        if self.settings.app_title:
            headers["X-Title"] = self.settings.app_title

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenRouterError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc.reason}") from exc


def _cmd_chat(args: argparse.Namespace) -> int:
    try:
        client = OpenRouterClient(settings_from_env(args.env, base_url=args.base_url, api_key=args.api_key))
        content = client.chat(
            [{"role": "user", "content": args.prompt}],
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            reasoning=_reasoning_from_args(args, base_url=client.settings.base_url),
        )
    except OpenRouterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(content)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenRouter inference helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat", help="send one chat prompt through OpenRouter")
    chat.add_argument("prompt")
    chat.add_argument("--env", type=Path, default=Path(".env"))
    chat.add_argument("--base-url")
    chat.add_argument("--api-key")
    chat.add_argument("--model")
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--max-tokens", type=int, default=1000)
    chat.add_argument("--reasoning-effort", choices=["xhigh", "high", "medium", "low", "minimal", "none"])
    chat.add_argument("--reasoning-max-tokens", type=int)
    chat.add_argument("--include-reasoning", action="store_true")
    chat.set_defaults(func=_cmd_chat)

    return parser


def _reasoning_from_args(args: argparse.Namespace, *, base_url: str | None = None) -> dict[str, object] | None:
    if args.reasoning_effort and args.reasoning_max_tokens is not None:
        raise OpenRouterError("use either --reasoning-effort or --reasoning-max-tokens, not both")
    if _uses_non_openrouter_base_url(base_url) and args.reasoning_max_tokens is None:
        return None
    if not args.reasoning_effort and args.reasoning_max_tokens is None:
        return None

    reasoning: dict[str, object] = {"exclude": not args.include_reasoning}
    if args.reasoning_effort:
        reasoning["effort"] = args.reasoning_effort
    else:
        reasoning["max_tokens"] = args.reasoning_max_tokens
    return reasoning


def _uses_non_openrouter_base_url(base_url: str | None) -> bool:
    resolved_base_url = (base_url or "").strip().rstrip("/")
    return bool(resolved_base_url and resolved_base_url != DEFAULT_BASE_URL)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
