"""LLM backends behind one interface.

    llm.stream(messages, tools, system, role="chat") -> iterator of TextDelta | ToolCall | Done
    llm.complete(prompt, system, role="extractor", json_schema=None) -> str

`messages` use a provider-neutral format:
    {"role": "user", "content": "text"}
    {"role": "assistant", "content": "text", "tool_calls": [{"id","name","input"}], "raw": <provider blocks>}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "text", "is_error": False}

`raw` holds the provider's exact assistant content (thinking blocks included) and is replayed
unchanged inside one turn's tool loop. Completed turns are stored text-only, which keeps every
request append-only from the API's point of view even though the system prompt (retrieved
memory, time) is rebuilt per turn.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """An error with a short, speakable message."""


class LLMConfigError(LLMError):
    pass


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class Done:
    stop_reason: str                 # end_turn | tool_use | max_tokens | refusal | cancelled
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None                  # provider content to replay within the turn
    usage: dict = field(default_factory=dict)
    model: str = ""


@dataclass
class SystemPrompt:
    static: str                      # persona + rules: identical every turn -> prompt-cached
    dynamic: str = ""                # time, retrieved facts and snippets

    def joined(self) -> str:
        return self.static + ("\n\n" + self.dynamic if self.dynamic else "")


class LLM(Protocol):
    name: str

    def model_for(self, role: str) -> str: ...

    def stream(self, messages: list[dict], tools: list[dict] | None, system: SystemPrompt,
               role: str = "chat", cancel: threading.Event | None = None) -> Iterator[Any]: ...

    def complete(self, prompt: str, system: str = "", role: str = "extractor",
                 json_schema: dict | None = None, max_tokens: int = 1500) -> str: ...


# ----------------------------------------------------------------- Anthropic --
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, cfg):
        try:
            import anthropic
        except ImportError as e:
            raise LLMConfigError("The 'anthropic' package is not installed. Run: pip install -r requirements.txt") from e
        self._anthropic = anthropic
        self.c = cfg.section("llm.anthropic")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and paste your key, "
                "or switch to a local model with llm.backend: ollama in config.yaml.")
        kw = {"base_url": self.c["base_url"]} if self.c.get("base_url") else {}
        self.client = anthropic.Anthropic(timeout=float(self.c.get("timeout_s", 30)), max_retries=2, **kw)

    def model_for(self, role: str) -> str:
        return {"chat": self.c.get("chat_model", "claude-haiku-5-5"),
                "think": self.c.get("think_model", "claude-opus-5-5"),
                "extractor": self.c.get("extractor_model", "claude-haiku-5-5")}.get(role, role)

    def _effort(self, role: str) -> str | None:
        return {"chat": self.c.get("chat_effort"), "think": self.c.get("think_effort"),
                "extractor": "low"}.get(role)

    @staticmethod
    def convert(messages: list[dict], strip_thinking: bool = False) -> list[dict]:
        out: list[dict] = []

        def add_user(block: dict) -> None:
            if out and out[-1]["role"] == "user":
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

        for m in messages:
            if m["role"] == "assistant":
                raw = m.get("raw")
                if raw is not None:
                    content = list(raw)
                    if strip_thinking:
                        content = [b for b in content
                                   if _block_type(b) not in ("thinking", "redacted_thinking")]
                else:
                    content = []
                    if m.get("content"):
                        content.append({"type": "text", "text": m["content"]})
                    for tc in m.get("tool_calls") or []:
                        content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"],
                                        "input": tc["input"]})
                if content:
                    out.append({"role": "assistant", "content": content})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"],
                         "content": str(m.get("content", ""))}
                if m.get("is_error"):
                    block["is_error"] = True
                add_user(block)
            else:
                add_user({"type": "text", "text": m["content"]})
        return out

    def _system_blocks(self, system: SystemPrompt) -> list[dict]:
        blocks = [{"type": "text", "text": system.static, "cache_control": {"type": "ephemeral"}}]
        if system.dynamic:
            blocks.append({"type": "text", "text": system.dynamic})
        return blocks

    def stream(self, messages, tools, system, role="chat", cancel=None):
        a = self._anthropic
        model = self.model_for(role)
        kwargs: dict[str, Any] = dict(model=model, max_tokens=int(self.c.get("max_tokens", 2048)),
                                      system=self._system_blocks(system))
        if tools:
            kwargs["tools"] = [dict(t, eager_input_streaming=True) for t in tools]
        effort = self._effort(role)
        if effort:
            kwargs["output_config"] = {"effort": effort}
        if self.c.get("thinking") == "disabled" and role != "think":
            kwargs["thinking"] = {"type": "disabled"}
        use_fallback = role == "think" and bool(self.c.get("think_fallbacks", True))

        strip = False
        for attempt in range(3):
            kwargs["messages"] = self.convert(messages, strip_thinking=strip)
            try:
                yield from self._stream_once(kwargs, use_fallback, cancel, model)
                return
            except ValueError as e:      # eager tool-input JSON the SDK could not parse
                log.warning("unparseable tool input, retrying turn: %s", e)
                continue
            except a.BadRequestError as e:
                if not strip and "signature" in str(e) and "thinking" in str(e):
                    log.warning("thinking block rejected; retrying once without thinking blocks")
                    strip = True
                    continue
                raise LLMError(f"The model rejected the request: {_short(e)}") from e
            except a.AuthenticationError as e:
                raise LLMConfigError("My Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env.") from e
            except a.NotFoundError as e:
                raise LLMConfigError(f"The model {model} was not found. Check config.yaml.") from e
            except a.RateLimitError as e:
                raise LLMError("I'm being rate limited right now. Try again in a moment.") from e
            except a.APIConnectionError as e:
                raise LLMError("I can't reach the Anthropic API. Check your internet connection.") from e
            except a.APIStatusError as e:
                raise LLMError(f"The Anthropic API returned an error ({e.status_code}).") from e
        raise LLMError("The model returned malformed tool input twice.")

    def _stream_once(self, kwargs, use_fallback, cancel, model):
        if use_fallback:
            ctx = self.client.beta.messages.stream(**kwargs, betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default")
        else:
            ctx = self.client.messages.stream(**kwargs)
        with ctx as stream:
            for event in stream:
                if cancel is not None and cancel.is_set():
                    yield Done("cancelled", model=model)
                    return
                if event.type == "text":
                    yield TextDelta(event.text)
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield Done("refusal", model=model)
            return
        text = "".join(b.text for b in final.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else {})
                 for b in final.content if b.type == "tool_use"]
        stop = final.stop_reason or "end_turn"
        if stop == "max_tokens" and calls:
            calls = []          # a truncated tool input must never run
        for c in calls:
            yield c
        u = final.usage
        usage = {"input": u.input_tokens, "output": u.output_tokens,
                 "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}
        yield Done(stop, text=text, tool_calls=calls, raw=list(final.content), usage=usage,
                   model=getattr(final, "model", model))

    def complete(self, prompt, system="", role="extractor", json_schema=None, max_tokens=1500):
        a = self._anthropic
        kwargs: dict[str, Any] = dict(model=self.model_for(role), max_tokens=max_tokens,
                                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        oc: dict[str, Any] = {"effort": self._effort(role) or "low"}
        if json_schema:
            oc["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs["output_config"] = oc
        try:
            resp = self.client.messages.create(**kwargs)
        except a.AuthenticationError as e:
            raise LLMConfigError("Anthropic API key rejected.") from e
        except a.APIError as e:
            raise LLMError(f"Anthropic request failed: {_short(e)}") from e
        if resp.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in resp.content if b.type == "text")


# -------------------------------------------------------------------- Ollama --
class OllamaLLM:
    name = "ollama"

    def __init__(self, cfg):
        try:
            import ollama
        except ImportError as e:
            raise LLMConfigError("The 'ollama' package is not installed. Run: pip install -r requirements.txt") from e
        self._ollama = ollama
        self.c = cfg.section("llm.ollama")
        self.host = self.c.get("host", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)

    def model_for(self, role: str) -> str:
        return {"chat": self.c.get("chat_model", "llama3.2:3b"),
                "think": self.c.get("think_model") or self.c.get("chat_model", "llama3.2:3b"),
                "extractor": self.c.get("extractor_model") or self.c.get("chat_model", "llama3.2:3b"),
                }.get(role, role)

    @staticmethod
    def convert(messages: list[dict], system: str) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": {"name": tc["name"], "arguments": tc["input"]}}
                                         for tc in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "content": str(m.get("content", "")), "tool_name": m.get("name", "")})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def _options(self) -> dict:
        return {"num_ctx": int(self.c.get("num_ctx", 8192))}

    def _wrap(self, e: Exception, model: str) -> LLMError:
        msg = str(e)
        if isinstance(e, self._ollama.ResponseError) and ("not found" in msg or getattr(e, "status_code", 0) == 404):
            return LLMConfigError(f"The local model {model} isn't installed. Run: ollama pull {model}")
        if isinstance(e, (ConnectionError, OSError)) or "connect" in msg.lower():
            return LLMConfigError(f"I can't reach Ollama at {self.host}. Start it with: ollama serve")
        return LLMError(f"Ollama error: {msg[:120]}")

    def stream(self, messages, tools, system, role="chat", cancel=None):
        model = self.model_for(role)
        otools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                    "parameters": t["input_schema"]}} for t in tools or []]
        text_parts: list[str] = []
        calls: list[ToolCall] = []
        try:
            resp = self.client.chat(model=model, messages=self.convert(messages, system.joined()),
                                    tools=otools or None, stream=True, options=self._options(),
                                    keep_alive=self.c.get("keep_alive", "30m"))
            for chunk in resp:
                if cancel is not None and cancel.is_set():
                    yield Done("cancelled", model=model)
                    return
                msg = chunk.message
                if msg.content:
                    text_parts.append(msg.content)
                    yield TextDelta(msg.content)
                for tc in msg.tool_calls or []:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append(ToolCall("call_" + uuid.uuid4().hex[:12], tc.function.name, dict(args or {})))
        except LLMError:
            raise
        except Exception as e:
            raise self._wrap(e, model) from e
        for c in calls:
            yield c
        yield Done("tool_use" if calls else "end_turn", text="".join(text_parts), tool_calls=calls,
                   raw=None, model=model)

    def complete(self, prompt, system="", role="extractor", json_schema=None, max_tokens=1500):
        model = self.model_for(role)
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        try:
            resp = self.client.chat(model=model, messages=msgs, format=json_schema or None,
                                    options=dict(self._options(), num_predict=max_tokens),
                                    keep_alive=self.c.get("keep_alive", "30m"))
        except Exception as e:
            raise self._wrap(e, model) from e
        return resp.message.content or ""


def _block_type(b: Any) -> str:
    return b.get("type", "") if isinstance(b, dict) else getattr(b, "type", "")


def _short(e: Exception) -> str:
    return str(e).replace("\n", " ")[:160]


def make_llm(cfg) -> LLM:
    backend = (cfg.get("llm.backend") or "anthropic").lower()
    if backend == "anthropic":
        return AnthropicLLM(cfg)
    if backend == "ollama":
        return OllamaLLM(cfg)
    raise LLMConfigError(f"Unknown llm.backend '{backend}'. Use 'anthropic' or 'ollama'.")
