import time

from bi.brain.agent import Brain, Say
from bi.brain.llm import ToolCall
from bi.tools import ToolInputError, build_registry, validate
from bi.tools.web import parse_duckduckgo
from tests.conftest import ScriptedLLM


def says(evs):
    return " ".join(e.text for e in evs if isinstance(e, Say))


def test_validate_schema_subset():
    schema = {"type": "object", "properties": {"seconds": {"type": "integer", "minimum": 1}, "label": {"type": "string"}},
              "required": ["seconds"]}
    assert validate(schema, {"seconds": "300", "label": "tea", "junk": 1}) == {"seconds": 300, "label": "tea"}
    for bad in ({}, {"seconds": 0}, {"seconds": "abc"}, {"seconds": True}, "nope"):
        try:
            validate(schema, bad)
            raise AssertionError(bad)
        except ToolInputError:
            pass


def test_registry_specs_and_config_enforcement(cfg, store):
    reg = build_registry(cfg, store)
    names = [s["name"] for s in reg.specs()]
    assert names == [n for n in cfg.get("tools.enabled") if n in names]       # stable config order
    assert {"get_time", "take_note", "set_timer", "web_search", "file_search"} <= set(names)
    cfg.set("tools.side_effects", ["set_timer"])                               # take_note no longer listed
    cfg.set("tools.network", [])
    reg2 = build_registry(cfg, store)
    assert "take_note" not in reg2.tools and "web_search" not in reg2.tools and "set_timer" in reg2.tools
    reg.close(); reg2.close()


def test_confirmation_flow_runs_tool_after_yes(cfg, store):
    llm = ScriptedLLM([[ToolCall("t1", "set_timer", {"seconds": 300, "label": "pasta"})],
                       "Done, your pasta timer is running."])
    reg = build_registry(cfg, store)
    b = Brain(cfg, store, llm, reg)
    q = says(b.respond("set a pasta timer for five minutes"))
    assert q == "Want me to set a pasta timer for 5 minutes?"
    assert store.pending_reminders() == []                                     # nothing ran yet
    out = says(b.respond("yes"))
    assert out == "Done, your pasta timer is running."
    rem = store.pending_reminders()
    assert len(rem) == 1 and rem[0]["kind"] == "timer" and rem[0]["text"] == "pasta"
    # the continuation replayed the exact turn: same system prompt, tool_use then tool_result
    msgs, _, system, _ = llm.calls[1]
    assert msgs[-2]["tool_calls"][0]["name"] == "set_timer" and msgs[-1]["role"] == "tool"
    assert system is llm.calls[0][2]
    log = store._q("SELECT tool, status, side_effect FROM tool_log")
    assert log == [("set_timer", "ok", 1)]
    reg.close()


def test_confirmation_declined(cfg, store):
    llm = ScriptedLLM([[ToolCall("t1", "take_note", {"text": "buy milk"})]])
    reg = build_registry(cfg, store)
    b = Brain(cfg, store, llm, reg)
    assert "take a note saying: buy milk" in says(b.respond("note buy milk"))
    assert says(b.respond("no")) == "Okay, I won't."
    assert store.notes() == [] and len(llm.calls) == 1
    reg.close()


def test_unconfirmed_new_request_drops_pending(cfg, store):
    llm = ScriptedLLM([[ToolCall("t1", "take_note", {"text": "x"})], "It's three o'clock."])
    reg = build_registry(cfg, store)
    b = Brain(cfg, store, llm, reg)
    list(b.respond("note x"))
    assert says(b.respond("what time is it")) == "It's three o'clock."
    assert store.notes() == [] and b.pending is None
    reg.close()


def test_read_only_tool_runs_without_confirmation(cfg, store, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "taxes-2025.md").write_text("Accountant meeting moved to March 3rd. Bring the W-2 forms.")
    (docs / "recipes.txt").write_text("Lasagna: 3 layers of pasta.")
    llm = ScriptedLLM([[ToolCall("t1", "file_search", {"query": "accountant meeting"})], "It's on March 3rd."])
    reg = build_registry(cfg, store)
    b = Brain(cfg, store, llm, reg)
    assert says(b.respond("when is my accountant meeting?")) == "It's on March 3rd."
    tool_msg = llm.calls[1][0][-1]
    assert tool_msg["role"] == "tool" and "taxes-2025.md" in tool_msg["content"] and "March 3rd" in tool_msg["content"]
    assert "recipes" not in tool_msg["content"]
    reg.close()


def test_tool_errors_become_tool_results(cfg, store):
    reg = build_registry(cfg, store)
    out, err = reg.execute("set_reminder", {"text": "call mom", "at": "not a time"})
    assert err and "can't parse" in out
    out, err = reg.execute("delete_everything", {})
    assert err and "Unknown" in out
    out, err = reg.execute("get_time", {"timezone": "Asia/Tokyo"})
    assert not err and "JST" in out
    reg.close()


def test_reminder_fires_spoken_alert(cfg, store):
    alerts = []
    reg = build_registry(cfg, store, on_alert=alerts.append)
    out, err = reg.execute("set_reminder", {"text": "stretch", "in_minutes": 0.01})
    assert not err and "stretch" in out
    reg.execute("set_timer", {"seconds": 1, "label": "tea"})
    deadline = time.time() + 4
    while len(alerts) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert alerts == ["Reminder: stretch.", "Your tea timer is done."]
    assert store.pending_reminders() == []
    reg.close()


def test_notes_roundtrip(cfg, store):
    reg = build_registry(cfg, store)
    reg.execute("take_note", {"text": "Call the plumber Tuesday"})
    reg.execute("take_note", {"text": "Ideas: hiking trip"})
    assert "plumber" in reg.execute("read_notes", {"query": "plumber"})[0]
    assert reg.execute("read_notes", {})[0].startswith("#2")
    reg.close()


def test_duckduckgo_parser():
    page = ('<div class="result results_links results_links_deep web-result"><a rel="nofollow" class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=x">Example <b>Title</b></a>'
            '<a class="result__snippet" href="#">Some <b>snippet</b> &amp; text</a></div>'
            '<div class="result results_links"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">Ad</a></div>')
    assert parse_duckduckgo(page) == [{"title": "Example Title", "snippet": "Some snippet & text",
                                       "url": "https://example.com/page"}]
