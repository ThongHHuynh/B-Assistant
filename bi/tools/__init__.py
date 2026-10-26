"""Tool registry for function calling.

Each tool has a JSON schema, a handler and flags. A tool flagged side_effect or network is only enabled
if config.yaml lists it under tools.side_effects / tools.network, and every call to such a tool is
logged to the tool_log table. Tools in tools.require_confirmation need a spoken "yes" first.
No destructive tools (deleting files, sending messages) exist.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[dict], str]
    side_effect: bool = False
    network: bool = False
    describe: Callable[[dict], str] | None = None     # "set a timer for 5 minutes" (for confirmations)

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "input_schema": self.schema}


class ToolInputError(ValueError):
    pass


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(schema: dict, data) -> dict:
    """Small JSON-schema subset check (type, required, enum, minimum). Model output is untrusted."""
    if not isinstance(data, dict):
        raise ToolInputError("input must be an object")
    props = schema.get("properties", {})
    for req in schema.get("required", []):
        if req not in data or data[req] in (None, ""):
            raise ToolInputError(f"missing required field '{req}'")
    out = {}
    for k, v in data.items():
        if k not in props:
            continue            # ignore extras instead of failing the turn
        p = props[k]
        t = p.get("type")
        if t == "integer" and isinstance(v, float) and v.is_integer():
            v = int(v)
        if t == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
            v = int(v)
        if t and not isinstance(v, _TYPES[t]) or (t in ("integer", "number") and isinstance(v, bool)):
            raise ToolInputError(f"'{k}' must be {t}")
        if "enum" in p and v not in p["enum"]:
            raise ToolInputError(f"'{k}' must be one of {p['enum']}")
        if "minimum" in p and v < p["minimum"]:
            raise ToolInputError(f"'{k}' must be >= {p['minimum']}")
        out[k] = v
    return out


class ToolRegistry:
    def __init__(self, cfg, store):
        t = cfg.section("tools")
        self.cfg = cfg
        self.store = store
        self.enabled_names = list(t.get("enabled", []))
        self.side_effects = set(t.get("side_effects", []))
        self.network = set(t.get("network", []))
        self.confirm = set(t.get("require_confirmation", []))
        self.tools: dict[str, Tool] = {}
        self.on_alert: Callable[[str], None] | None = None
        self._closers: list[Callable[[], None]] = []

    def register(self, tool: Tool) -> None:
        if tool.name not in self.enabled_names:
            return
        if tool.side_effect and tool.name not in self.side_effects:
            log.warning("tool %s changes state but isn't listed in tools.side_effects; disabled", tool.name)
            return
        if tool.network and tool.name not in self.network:
            log.warning("tool %s uses the network but isn't listed in tools.network; disabled", tool.name)
            return
        self.tools[tool.name] = tool

    def specs(self) -> list[dict]:
        # deterministic order (config order) keeps the tools block of the prompt cacheable
        return [self.tools[n].spec() for n in self.enabled_names if n in self.tools]

    def requires_confirmation(self, name: str) -> bool:
        return name in self.confirm and name in self.tools

    def confirmation_prompt(self, calls) -> str:
        parts = []
        for c in calls:
            tool = self.tools.get(c.name)
            if tool and tool.describe and self.requires_confirmation(c.name):
                try:
                    parts.append(tool.describe(validate(tool.schema, c.input)))
                except ToolInputError:
                    parts.append(c.name.replace("_", " "))
        return "Want me to " + " and ".join(parts or ["do that"]) + "?"

    def execute(self, name: str, data: dict) -> tuple[str, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown or disabled tool: {name}", True
        audited = tool.side_effect or tool.network
        try:
            args = validate(tool.schema, data)
            result = tool.handler(args)
            status, is_error = "ok", False
        except ToolInputError as e:
            result, status, is_error = f"Invalid input: {e}", "invalid", True
        except Exception as e:  # noqa: BLE001 - a failing tool must not crash the turn
            log.exception("tool %s failed", name)
            result, status, is_error = f"Tool error: {e}", "error", True
        if audited:
            log.info("tool %s(%s) -> %s", name, json.dumps(data)[:200], status)
            self.store.log_tool(name, data, result, status, tool.side_effect, tool.network)
        return result, is_error

    def set_alert(self, cb: Callable[[str], None]) -> None:
        self.on_alert = cb

    def alert(self, message: str) -> None:
        if self.on_alert:
            self.on_alert(message)
        else:
            log.info("alert (no listener): %s", message)

    def add_closer(self, fn: Callable[[], None]) -> None:
        self._closers.append(fn)

    def close(self) -> None:
        for fn in self._closers:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


def build_registry(cfg, store, on_alert=None) -> ToolRegistry:
    from .builtin import register_builtin
    reg = ToolRegistry(cfg, store)
    if on_alert:
        reg.set_alert(on_alert)
    register_builtin(reg, cfg, store)
    return reg
