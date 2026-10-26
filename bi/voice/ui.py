"""Console status line and single-key hotkeys (m = mute, q = quit, Enter = push-to-talk)."""
from __future__ import annotations

import os
import queue
import sys
import threading

ICONS = {"idle": "○ IDLE (say the wake word)", "listening": "● LISTENING", "thinking": "◐ THINKING",
         "speaking": "♪ SPEAKING", "follow_up": "◌ FOLLOW-UP (just talk)", "muted": "✕ MUTED (press m)"}


class StatusLine:
    def __init__(self, name: str = "Bi", enabled: bool = True, ptt: bool = False):
        self.name = name
        self.enabled = enabled and sys.stdout.isatty()
        self.ptt = ptt
        self.text = ""
        self.lock = threading.Lock()
        if os.name == "nt":
            os.system("")          # enables ANSI escape handling on Windows 10+

    def set(self, state: str, extra: str = "") -> None:
        label = ICONS.get(state, state.upper())
        if self.ptt and state == "idle":
            label = "○ IDLE (press Enter to talk)"
        self.text = f"[{self.name}] {label}" + (f"  {extra}" if extra else "")
        self._render()

    def _render(self) -> None:
        if not self.enabled:
            return
        with self.lock:
            sys.stdout.write("\r\x1b[2K" + self.text + "   (m: mute, q: quit)")
            sys.stdout.flush()

    def print(self, line: str) -> None:
        with self.lock:
            if self.enabled:
                sys.stdout.write("\r\x1b[2K")
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        self._render()


class KeyReader(threading.Thread):
    """Reads single keypresses without Enter (cbreak) into self.keys. No-op if stdin isn't a TTY."""

    def __init__(self):
        super().__init__(daemon=True, name="keys")
        self.keys: "queue.Queue[str]" = queue.Queue()
        self._old = None
        self.active = sys.stdin.isatty()

    def run(self) -> None:
        if not self.active:
            return
        if os.name == "nt":
            import msvcrt
            while True:
                ch = msvcrt.getwch()
                self.keys.put("\n" if ch == "\r" else ch.lower())
        else:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break
                    self.keys.put(ch.lower())
            finally:
                self.restore()

    def restore(self) -> None:
        if self._old is not None and os.name != "nt":
            import termios
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            except Exception:  # noqa: BLE001
                pass
            self._old = None

    def get(self) -> str | None:
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None
