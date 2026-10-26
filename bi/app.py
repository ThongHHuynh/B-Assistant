"""Wires config, memory, LLM, tools and brain together."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .brain.agent import Brain
from .brain.llm import LLMConfigError, make_llm
from .logs import setup_logging
from .memory.store import MemoryStore, open_store

log = logging.getLogger(__name__)


@dataclass
class App:
    cfg: object
    store: MemoryStore
    llm: object
    brain: Brain
    tools: object
    llm_error: str = ""

    def close(self) -> None:
        if self.tools is not None and hasattr(self.tools, "close"):
            self.tools.close()
        self.brain.close()
        self.store.close()


def build_app(cfg, on_alert=None, quiet: bool = False) -> App:
    setup_logging(cfg.path_of("paths.log_file", "data/logs/bi.log"), console=not quiet)
    store = open_store(cfg)
    llm, err = None, ""
    try:
        llm = make_llm(cfg)
    except LLMConfigError as e:
        err = str(e)
        log.error("LLM unavailable: %s", e)
    tools = None
    try:
        from .tools import build_registry
        tools = build_registry(cfg, store, on_alert=on_alert)
    except ImportError:
        pass
    brain = Brain(cfg, store, llm, tools)
    return App(cfg, store, llm, brain, tools, err)
