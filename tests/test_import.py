import json

from bi.import_history import import_conversations, load_conversations
from tests.conftest import ScriptedLLM
from bi.memory.extractor import FactExtractor

CHATGPT = [{"title": "Trip planning", "create_time": 1700000000, "current_node": "c",
            "mapping": {"a": {"message": None, "parent": None},
                        "b": {"parent": "a", "message": {"author": {"role": "user"}, "create_time": 1700000001,
                                                          "content": {"parts": ["I'm vegetarian, plan a dinner in Rome"]}}},
                        "x": {"parent": "b", "message": {"author": {"role": "assistant"}, "content": {"parts": ["(abandoned branch)"]}}},
                        "c": {"parent": "b", "message": {"author": {"role": "assistant"}, "create_time": 1700000002,
                                                          "content": {"parts": ["Try cacio e pepe."]}}}}}]
CLAUDE = [{"name": "Garden", "created_at": "2024-05-01T10:00:00Z",
           "chat_messages": [{"sender": "human", "text": "My tomatoes keep splitting", "created_at": "2024-05-01T10:00:00Z"},
                             {"sender": "assistant", "text": "Water more evenly."}]}]
MD = """# Monday chat
User: I run every morning at 6.
Assistant: Great habit!

---
## User
my cat is called Miso
## Assistant
Cute name.
"""


def test_parse_formats(tmp_path):
    (tmp_path / "g.json").write_text(json.dumps(CHATGPT))
    (tmp_path / "c.json").write_text(json.dumps(CLAUDE))
    (tmp_path / "m.md").write_text(MD)
    g = load_conversations(tmp_path / "g.json")
    assert [r for r, _, _ in g[0].turns] == ["user", "assistant"]
    assert g[0].turns[1][1] == "Try cacio e pepe."          # follows current_node, not the abandoned branch
    c = load_conversations(tmp_path / "c.json")
    assert c[0].title == "Garden" and c[0].turns[0] == ("user", "My tomatoes keep splitting", c[0].started_at)
    m = load_conversations(tmp_path / "m.md")
    assert len(m) == 2 and m[0].turns[0][1] == "I run every morning at 6." and m[1].turns[0][1] == "my cat is called Miso"


def test_import_runs_extractor(tmp_path, store):
    (tmp_path / "g.json").write_text(json.dumps(CHATGPT))
    convs = load_conversations(tmp_path / "g.json")
    llm = ScriptedLLM(completions=[json.dumps({"facts": [
        {"text": "User is vegetarian.", "category": "preference", "confidence": 0.9, "supersedes": []}]})])
    stats = import_conversations(store, convs, FactExtractor(store, llm), progress=lambda *_: None)
    assert stats == {"conversations": 1, "turns": 2, "chunks": 1, "facts": 1}
    assert store.list_facts()[0].text == "User is vegetarian."
    assert store.session(store.sessions()[0]["id"])["source"] == "import"
