"""Synchronous OpenAI-compatible client for local or hosted chat models.

This module is adapted from ``/home/xiaoyue/GRACE/common/local_llm.py`` and
kept intentionally lightweight for notebook and script use inside LiteSemRAG.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_PROVIDER = "local"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_API_KEY = "dummy"
DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_API_KEY_FILE = "API_KEY"

ALLOWED_CHAT_PARAMS = {
    "temperature",
    "max_tokens",
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "stream",
    "seed",
    "max_completion_tokens",
    "reasoning_effort",
    "verbosity",
    "response_format",
}

Message = dict[str, str]


@dataclass(frozen=True)
class LocalLLMConfig:
    """Configuration for local or hosted OpenAI-compatible chat completions."""

    provider: str = DEFAULT_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = DEFAULT_API_KEY
    api_key_file: str | None = None
    timeout: float | None = 120.0

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
        api_key_file: str | None = None,
    ) -> "LocalLLMConfig":
        provider_name = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER)
        provider_name = provider_name.strip().lower()
        if provider_name not in {"local", "openai"}:
            raise ValueError(
                f"Unsupported LLM provider {provider_name!r}; use 'local' or 'openai'."
            )

        timeout_raw = (
            os.getenv("OPENAI_TIMEOUT")
            if provider_name == "openai"
            else os.getenv("LOCAL_LLM_TIMEOUT")
        )
        timeout = float(timeout_raw) if timeout_raw else 120.0

        if provider_name == "openai":
            key_file = (
                api_key_file
                or os.getenv("OPENAI_API_KEY_FILE")
                or DEFAULT_OPENAI_API_KEY_FILE
            )
            api_key = os.getenv("OPENAI_API_KEY") or _read_api_key_file(key_file)
            return cls(
                provider=provider_name,
                base_url=os.getenv("OPENAI_BASE_URL", ""),
                model=model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
                api_key=api_key,
                api_key_file=key_file,
                timeout=timeout,
            )

        return cls(
            provider=provider_name,
            base_url=os.getenv("LOCAL_LLM_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.getenv("LOCAL_LLM_MODEL", DEFAULT_MODEL),
            api_key=os.getenv("LOCAL_LLM_API_KEY", DEFAULT_API_KEY),
            timeout=timeout,
        )


def _read_api_key_file(path: str | os.PathLike[str]) -> str:
    key_path = Path(path).expanduser()
    if not key_path.is_absolute():
        key_path = Path.cwd() / key_path
    text = key_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"API key file is empty: {key_path}")

    key_match = re.search(r"sk-[A-Za-z0-9_-]+", text)
    if key_match:
        return key_match.group(0)

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, value = line.split("=", 1)
            name = name.strip().strip("'\"")
            value = value.strip().strip(",").strip().strip("'\"")
            if name in {"OPENAI_API_KEY", "API_KEY", "chatgpt_api"}:
                return value
            if value.startswith("sk-"):
                return value
            continue
        if ":" in line:
            name, value = line.split(":", 1)
            name = name.strip().strip("'\"")
            value = value.strip().strip(",").strip().strip("'\"")
            if name in {"OPENAI_API_KEY", "API_KEY", "chatgpt_api"}:
                return value
            if value.startswith("sk-"):
                return value
        return line.strip().strip("'\"")

    raise ValueError(f"No API key found in {key_path}")


def _load_openai_class():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required for local_llm.py. Install it in "
            "the llm_graph environment before calling the local model."
        ) from exc
    return OpenAI


def _normalize_history(
    history_messages: Iterable[Mapping[str, Any]] | None,
) -> list[Message]:
    if not history_messages:
        return []

    normalized: list[Message] = []
    for idx, message in enumerate(history_messages):
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"history_messages[{idx}] has unsupported role: {role!r}")
        if content is None:
            content = ""
        normalized.append({"role": role, "content": str(content)})
    return normalized


def build_messages(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: Iterable[Mapping[str, Any]] | None = None,
) -> list[Message]:
    messages: list[Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(_normalize_history(history_messages))
    messages.append({"role": "user", "content": prompt})
    return messages


def _filter_chat_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if k in ALLOWED_CHAT_PARAMS}


def _chat_kwargs_for_model(model: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    filtered = _filter_chat_kwargs(kwargs)
    model_name = model.lower()
    if model_name.startswith(("gpt-5", "o1", "o3", "o4")):
        max_tokens = filtered.pop("max_tokens", None)
        if max_tokens is not None and "max_completion_tokens" not in filtered:
            filtered["max_completion_tokens"] = max_tokens
        filtered.pop("temperature", None)
        filtered.pop("top_p", None)
    return filtered


def _record_usage(response: Any, current_total: int) -> int:
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if isinstance(total_tokens, int):
        return current_total + total_tokens
    return current_total


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content or ""


class LocalLLMSyncClient:
    """Synchronous client for local or hosted OpenAI-compatible chat endpoints."""

    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig.from_env()
        OpenAI = _load_openai_class()
        client_kwargs: dict[str, Any] = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
        }
        if self.config.provider == "local":
            client_kwargs["base_url"] = self.config.base_url
        elif self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
        self.client = OpenAI(**client_kwargs)
        self.total_tokens = 0

    def chat(self, messages: list[Message], **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            **_chat_kwargs_for_model(self.config.model, kwargs),
        )
        self.total_tokens = _record_usage(response, self.total_tokens)
        return _response_text(response)

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Iterable[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        messages = build_messages(prompt, system_prompt, history_messages)
        return self.chat(messages, **kwargs)


class LocalLLMClient:
    """Compatibility facade exposing synchronous local LLM calls."""

    def __init__(self, config: LocalLLMConfig | None = None):
        self.config = config or LocalLLMConfig.from_env()
        self._sync_client: LocalLLMSyncClient | None = None

    @property
    def sync_client(self) -> LocalLLMSyncClient:
        if self._sync_client is None:
            self._sync_client = LocalLLMSyncClient(self.config)
        return self._sync_client

    @property
    def total_tokens(self) -> int:
        return self._sync_client.total_tokens if self._sync_client else 0

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Iterable[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        return self.sync_client.complete(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )


_DEFAULT_CLIENT: LocalLLMClient | None = None


def get_default_client() -> LocalLLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LocalLLMClient()
    return _DEFAULT_CLIENT


def complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: Iterable[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    return get_default_client().complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


def sync_llm_model_func(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: Iterable[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> str:
    return complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
