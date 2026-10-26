"""Conversation state machine (pure logic, no audio).

IDLE -> LISTENING -> THINKING -> SPEAKING -> FOLLOW_UP -> (speech) LISTENING ... -> (timeout) IDLE
"""
from __future__ import annotations

import enum
import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)


class State(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOW_UP = "follow_up"
    MUTED = "muted"


S = State
TRANSITIONS: dict[tuple[State, str], State] = {
    (S.IDLE, "wake"): S.LISTENING,
    (S.IDLE, "ptt"): S.LISTENING,
    (S.LISTENING, "speech_end"): S.THINKING,
    (S.LISTENING, "timeout"): S.IDLE,
    (S.LISTENING, "empty"): S.FOLLOW_UP,        # heard sound but no words: stay attentive briefly
    (S.THINKING, "first_audio"): S.SPEAKING,
    (S.THINKING, "done"): S.FOLLOW_UP,          # finished without speaking (e.g. "stop")
    (S.THINKING, "barge_in"): S.LISTENING,
    (S.SPEAKING, "done"): S.FOLLOW_UP,
    (S.SPEAKING, "barge_in"): S.LISTENING,
    (S.SPEAKING, "stop"): S.FOLLOW_UP,
    (S.THINKING, "stop"): S.FOLLOW_UP,
    (S.FOLLOW_UP, "speech_start"): S.LISTENING,
    (S.FOLLOW_UP, "wake"): S.LISTENING,
    (S.FOLLOW_UP, "ptt"): S.LISTENING,
    (S.FOLLOW_UP, "timeout"): S.IDLE,
    (S.MUTED, "unmute"): S.IDLE,
}
# events valid from any state
GLOBAL: dict[str, State] = {"sleep": S.IDLE, "cancel": S.IDLE, "mute": S.MUTED, "error": S.IDLE}


class StateMachine:
    def __init__(self, initial: State = State.IDLE, on_change: Callable[[State, State, str], None] | None = None):
        self.state = initial
        self.on_change = on_change
        self.entered_at = time.monotonic()
        self.history: list[tuple[State, str, State]] = []
        self._lock = threading.RLock()

    def can(self, event: str) -> bool:
        return (self.state, event) in TRANSITIONS or (event in GLOBAL and not (self.state == S.MUTED and event != "mute"))

    def fire(self, event: str) -> bool:
        with self._lock:
            old = self.state
            if (old, event) in TRANSITIONS:
                new = TRANSITIONS[(old, event)]
            elif event in GLOBAL and (old != S.MUTED or event == "mute"):
                new = GLOBAL[event]
            else:
                log.debug("ignored event %s in %s", event, old.name)
                return False
            self.state = new
            self.entered_at = time.monotonic()
            self.history.append((old, event, new))
            if len(self.history) > 200:
                self.history = self.history[-100:]
        log.info("state %s --%s--> %s", old.name, event, new.name)
        if self.on_change:
            self.on_change(old, new, event)
        return True

    def time_in_state(self) -> float:
        return time.monotonic() - self.entered_at
