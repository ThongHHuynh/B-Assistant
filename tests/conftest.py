import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bi.brain.llm import Done, TextDelta, ToolCall  # noqa: E402
from bi.config import load_config  # noqa: E402
from bi.memory.embed import HashEmbedder  # noqa: E402
from bi.memory.store import MemoryStore  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


class ScriptedLLM:
    """Test double for the LLM interface. Each scripted reply is a str or a list of events."""

    name = "scripted"

    def __init__(self, replies=None, completions=None):
        self.replies = list(replies or [])
        self.completions = list(completions or [])
        self.calls = []          # (messages, tools, system, role)

    def model_for(self, role):
        return f"fake-{role}"

    def stream(self, messages, tools, system, role="chat", cancel=None):
        self.calls.append((list(messages), tools, system, role))
        reply = self.replies.pop(0) if self.replies else "Okay."
        if isinstance(reply, str):
            reply = [reply]
        text, calls = "", []
        for item in reply:
            if isinstance(item, str):
                for i in range(0, len(item), 7):          # token-ish deltas
                    if cancel is not None and cancel.is_set():
                        yield Done("cancelled", model=self.model_for(role))
                        return
                    text += item[i:i + 7]
                    yield TextDelta(item[i:i + 7])
            elif isinstance(item, ToolCall):
                calls.append(item)
                yield item
        yield Done("tool_use" if calls else "end_turn", text=text, tool_calls=calls, raw=None,
                   model=self.model_for(role))

    def complete(self, prompt, system="", role="extractor", json_schema=None, max_tokens=1500):
        self.calls.append(([{"role": "user", "content": prompt}], None, system, role))
        return self.completions.pop(0) if self.completions else '{"facts": []}'


@pytest.fixture
def cfg(tmp_path):
    c = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    c.set("paths.db", str(tmp_path / "bi.db"))
    c.set("paths.data_dir", str(tmp_path))
    c.set("paths.log_file", str(tmp_path / "bi.log"))
    c.set("memory.embeddings.backend", "hash")
    c.set("tools.file_search.folders", [str(tmp_path / "docs")])
    return c


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "bi.db", HashEmbedder())
    yield s
    s.close()


@pytest.fixture
def fake_llm():
    return ScriptedLLM()
