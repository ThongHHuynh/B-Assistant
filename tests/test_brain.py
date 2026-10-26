import threading

from bi.brain.agent import Action, Brain, Say, to_second_person, to_third_person
from bi.brain.llm import ToolCall
from tests.conftest import ScriptedLLM


def run(brain, text, cancel=None):
    return list(brain.respond(text, cancel=cancel))


def says(events):
    return " ".join(e.text for e in events if isinstance(e, Say))


def test_chat_turn_streams_sentences_and_logs(cfg, store):
    llm = ScriptedLLM(["Hello there! It's nice to meet you. How can I help?"])
    b = Brain(cfg, store, llm)
    ev = run(b, "hi")
    sentences = [e.text for e in ev if isinstance(e, Say)]
    assert sentences == ["Hello there!", "It's nice to meet you.", "How can I help?"]
    turns = store.session_turns(b.session_id)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].text.startswith("Hello there!")
    # system prompt: static first (cacheable), dynamic has date/time
    system = llm.calls[0][2]
    assert "You are Bi" in system.static and "Current date and time" in system.dynamic


def test_history_is_text_only_and_bounded(cfg, store):
    cfg.set("llm.max_history_turns", 2)
    llm = ScriptedLLM([f"Answer {i}." for i in range(5)])
    b = Brain(cfg, store, llm)
    for i in range(5):
        run(b, f"question {i}")
    msgs = llm.calls[-1][0]
    assert msgs[-1] == {"role": "user", "content": "question 4"}
    assert len(msgs) == 5          # 2 previous pairs + current
    assert all("raw" not in m for m in msgs)


def test_stop_and_sleep_bypass_llm(cfg, store):
    llm = ScriptedLLM()
    b = Brain(cfg, store, llm)
    assert [e for e in run(b, "stop") if isinstance(e, Action)][0].kind == "stop"
    ev = run(b, "go to sleep")
    assert any(isinstance(e, Action) and e.kind == "sleep" for e in ev)
    assert any(isinstance(e, Action) and e.kind == "cancel" for e in run(b, "never mind"))
    assert llm.calls == []


def test_remember_and_recall_and_forget(cfg, store):
    llm = ScriptedLLM()
    b = Brain(cfg, store, llm)
    out = says(run(b, "Remember that my sister's name is Anna"))
    assert "your sister's name is Anna" in out
    facts = store.list_facts()
    assert len(facts) == 1 and facts[0].text == "User's sister's name is Anna." and facts[0].explicit
    assert facts[0].category == "person"
    assert "sister" in says(run(b, "what do you know about me?"))
    # forget asks for confirmation first
    out = says(run(b, "forget my sister's name"))
    assert "forget" in out.lower() and "Should I go ahead" in out
    assert store.list_facts()                     # not yet deleted
    assert "forgotten" in says(run(b, "yes"))
    assert store.list_facts(include_inactive=True) == []
    # the raw log no longer contains it anywhere (source turn, Bi's echo, recall and confirmation turns)
    assert all("Anna" not in t.text for t in store.session_turns(b.session_id))


def test_forget_declined_keeps_fact(cfg, store):
    b = Brain(cfg, store, ScriptedLLM())
    run(b, "remember that I like green tea")
    run(b, "forget green tea")
    assert "keep" in says(run(b, "no"))
    assert len(store.list_facts()) == 1


def test_think_harder_routes_one_turn(cfg, store):
    llm = ScriptedLLM(["Deep answer.", "Quick answer."])
    b = Brain(cfg, store, llm)
    run(b, "think harder, why is the sky blue?")
    run(b, "and grass?")
    assert llm.calls[0][3] == "think" and llm.calls[0][0][-1]["content"] == "why is the sky blue?"
    assert llm.calls[1][3] == "chat"


def test_cancel_mid_stream_marks_interrupted(cfg, store):
    cancel = threading.Event()
    llm = ScriptedLLM(["This is a long answer that will be cut off. Second sentence here."])
    b = Brain(cfg, store, llm)
    out = []
    for e in b.respond("tell me a story", cancel=cancel):
        out.append(e)
        cancel.set()
    assert store.session_turns(b.session_id)[-1].text.endswith("[interrupted]")


def test_no_llm_still_handles_memory(cfg, store):
    b = Brain(cfg, store, None)
    assert "language model" in says(run(b, "what's the weather"))
    assert "remember" in says(run(b, "remember that I live in Lisbon")).lower()


def test_secret_redacted_from_raw_log(cfg, store):
    b = Brain(cfg, store, ScriptedLLM(["Noted."]))
    run(b, "my card is 4111 1111 1111 1111 ok")
    user = [t for t in store.session_turns(b.session_id) if t.role == "user"][0]
    assert "4111" not in user.text and "[redacted card number]" in user.text


def test_pronoun_helpers():
    assert to_third_person("I prefer tea and my dog is Rex") == "User prefer tea and User's dog is Rex"
    assert to_second_person("User prefers tea.") == "you prefer tea"
    assert to_second_person("User's sister is Anna.") == "your sister is Anna"
