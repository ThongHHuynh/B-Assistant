import json
import time

import pytest

from bi.memory.consolidate import consolidate, decay, merge_duplicates
from bi.memory.extractor import FactExtractor, _parse
from bi.memory.privacy import allowed_to_store, contains_secret, redact_secrets, sensitive_category
from bi.memory.retrieval import Retriever, rank_facts
from bi.memory.store import Fact, MemoryStore
from tests.conftest import ScriptedLLM


# ------------------------------------------------------------ insert/supersede/forget --
def test_insert_and_dedupe(store):
    fid, created = store.add_fact("User prefers tea over coffee", "preference", 0.7)
    assert created and store.get_fact(fid).text == "User prefers tea over coffee."
    fid2, created2 = store.add_fact("User prefers tea over coffee.", "preference", 0.8)
    assert fid2 == fid and not created2
    assert store.get_fact(fid).confidence >= 0.8          # reinforced, not duplicated
    assert len(store.list_facts()) == 1


def test_supersede_keeps_history(store):
    old, _ = store.add_fact("User lives in Berlin", "other", 0.9)
    new, _ = store.add_fact("User lives in Lisbon", "other", 0.9, supersedes=[old])
    assert [f.id for f in store.list_facts()] == [new]
    o = store.get_fact(old)
    assert o.superseded_by == new and not o.active
    assert {f.id for f in store.list_facts(include_inactive=True)} == {old, new}
    # superseded facts never come back from search
    assert all(f.id != old for f, _ in store.search_facts("where does the user live", k=5))


def test_forget_deletes_fully_with_history(store):
    sid = store.start_session()
    tid = store.add_turn(sid, "user", "I live in Berlin now")
    old, _ = store.add_fact("User lives in Berlin", "other", 0.9, source_turn_id=tid)
    new, _ = store.add_fact("User lives in Lisbon", "other", 0.9, supersedes=[old])
    store.add_chunk("turns", sid, "User: I live in Berlin now", tid, tid)
    deleted = store.delete_fact(new, cascade=True, scrub_sources=True)
    assert set(deleted) == {old, new}
    assert store.list_facts(include_inactive=True) == []
    assert store.index.get("fact", old) is None and store.index.get("fact", new) is None
    assert "Berlin" not in store.turn(tid).text
    assert store.search_chunks("Berlin", k=3) == []


def test_delete_without_cascade_reactivates_nothing_wrong(store):
    a, _ = store.add_fact("User drives a Honda", "other", 0.8)
    b, _ = store.add_fact("User drives a Tesla", "other", 0.8, supersedes=[a])
    store.delete_fact(b, cascade=False)
    # the older fact is un-superseded so the row is no longer dangling
    assert store.get_fact(a).superseded_by is None


def test_turns_and_sessions(store):
    s1 = store.start_session()
    store.add_turn(s1, "user", "hello")
    store.add_turn(s1, "assistant", "hi")
    s2 = store.start_session()
    store.add_turn(s2, "user", "again")
    assert [t.text for t in store.recent_turns(5, exclude_session=s2)] == ["hello", "hi"]
    assert store.stats()["turns"] == 3


def test_vector_backends_agree(tmp_path):
    from bi.memory.embed import HashEmbedder
    results = {}
    for backend in ("sqlite-vec", "numpy"):
        s = MemoryStore(tmp_path / f"{backend}.db", HashEmbedder(), backend)
        for t in ["User loves hiking in the mountains", "User's dog is named Rex", "User works as a nurse"]:
            s.add_fact(t)
        results[backend] = [f.text for f, _ in s.search_facts("what is my dog called", k=3)]
        assert s.index.backend == backend
        s.close()
    assert results["sqlite-vec"][0] == results["numpy"][0] == "User's dog is named Rex."


# ------------------------------------------------------------------- retrieval --
def _f(i, conf=0.7, age_days=0.0, active=True, explicit=False):
    now = time.time()
    return Fact(i, f"fact {i}", "other", None, now, now - age_days * 86400, conf, None if active else 99,
                explicit, False, None, 0)


def test_rank_facts_blends_similarity_confidence_recency():
    now = time.time()
    ranked = rank_facts([(_f(1, conf=0.3), 0.60), (_f(2, conf=0.95), 0.60), (_f(3), 0.90),
                         (_f(4), 0.10), (_f(5, active=False), 0.99), (_f(6, age_days=300), 0.61)],
                        now, min_score=0.3, k=10)
    ids = [f.id for f, _ in ranked]
    assert ids[0] == 3                       # most similar wins
    assert ids.index(2) < ids.index(1)       # same similarity: higher confidence first
    assert 4 not in ids and 5 not in ids     # below threshold / superseded are dropped


def test_retriever_end_to_end_and_budget(cfg, store):
    for t in ["User's sister is named Anna", "User is allergic to peanuts", "User plays guitar on weekends",
              "User works on a robotics startup"]:
        store.add_fact(t, "other", 0.9)
    sid = store.start_session()
    store.add_turn(sid, "user", "earlier session question")
    store.add_chunk("turns", sid, "User: we talked about the guitar amp\nBi: a tube amp sounds warm")
    r = Retriever(store, cfg)
    res = r.retrieve("what's my sister called", session_id=None)
    assert res.facts and res.facts[0].text == "User's sister is named Anna."
    assert [t.text for t in res.recent] == ["earlier session question"]
    assert res.elapsed_ms < 100 and not res.timed_out
    res2 = r.retrieve("guitar amp", session_id="other")
    assert any("guitar" in c["text"] for c in res2.snippets)
    # a zero budget degrades gracefully instead of blocking
    res3 = r.retrieve("anything", session_id=None, budget_ms=0)
    assert res3.elapsed_ms < 50


def test_semantic_retrieval_with_minilm(tmp_path, cfg):
    """Needs the real MiniLM model (ONNX); skipped if it can't be downloaded."""
    from bi.memory.embed import OnnxMiniLMEmbedder
    try:
        emb = OnnxMiniLMEmbedder(cfg.get("memory.embeddings.onnx_url"), cfg.root / "models")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"MiniLM unavailable: {e}")
    s = MemoryStore(tmp_path / "m.db", emb)
    for t in ["User's favorite drink is a flat white", "User's sister is named Anna",
              "User is training for a marathon", "User works as a backend engineer"]:
        s.add_fact(t)
    top = lambda q: s.search_facts(q, k=1)[0][0].text  # noqa: E731
    assert "flat white" in top("what coffee should you order for me?")
    assert "marathon" in top("how is my running going")
    assert "engineer" in top("what do I do for a living")
    s.close()


# ------------------------------------------------------------------- extractor --
def test_extractor_adds_supersedes_and_filters(cfg, store):
    old, _ = store.add_fact("User lives in Berlin", "other", 0.9)
    sid = store.start_session()
    store.add_turn(sid, "user", "I just moved to Lisbon. My password is hunter2. I'm seeing a therapist.")
    store.add_turn(sid, "assistant", "Nice!")
    store.add_turn(sid, "user", "Also I really prefer window seats.")
    reply = json.dumps({"facts": [
        {"text": "User lives in Lisbon.", "category": "other", "confidence": 0.9, "supersedes": [old]},
        {"text": "User's password is hunter2.", "category": "other", "confidence": 0.9, "supersedes": []},
        {"text": "User is seeing a therapist.", "category": "other", "confidence": 0.8, "supersedes": []},
        {"text": "User prefers window seats.", "category": "preference", "confidence": 0.9, "supersedes": [999]},
    ]})
    ex = FactExtractor(store, ScriptedLLM(completions=[reply]))
    added = ex.run(sid)
    texts = {f.text for f in store.list_facts()}
    assert texts == {"User lives in Lisbon.", "User prefers window seats."}
    assert len(added) == 2 and store.get_fact(old).superseded_by is not None
    seat = next(f for f in store.list_facts() if "window" in f.text)
    assert store.turn(seat.source_turn_id).text.startswith("Also I really prefer")
    assert ex.run(sid) == []                 # already extracted up to the last turn


def test_extractor_parse_tolerates_noise():
    assert _parse("Sure! {\"facts\": [{\"text\": \"User likes jazz\", \"category\": \"preference\", "
                  "\"confidence\": \"0.8\", \"supersedes\": [\"3\"]}]} done") == [
        {"text": "User likes jazz", "category": "preference", "confidence": 0.8, "supersedes": [3]}]
    assert _parse("not json") == []


# --------------------------------------------------------------------- privacy --
@pytest.mark.parametrize("text", ["my password is hunter2", "card 4111 1111 1111 1111",
                                  "key sk-ant-api03-abcdefghijklmnop", "SSN 123-45-6789",
                                  "my passport number is X1234567"])
def test_secrets_detected(text):
    assert contains_secret(text)
    assert not allowed_to_store(text, explicit=False)[0]
    assert allowed_to_store(text, explicit=True)[0]


def test_non_secrets_and_sensitive():
    assert not contains_secret("I have 2 kids and a 2019 car")
    assert not contains_secret("call me at 4pm about order 1234")
    assert sensitive_category("I was diagnosed with diabetes") == "health"
    assert sensitive_category("I like pizza") is None
    assert redact_secrets("pin: 4455 ok") == "[redacted password] ok"


# ---------------------------------------------------------------- consolidation --
def test_consolidation_merges_decays_and_summarizes(cfg, store):
    a, _ = store.add_fact("User likes jazz music", "preference", 0.6)
    b, _ = store.add_fact("User likes jazz music a lot", "preference", 0.9, supersedes=[0])
    assert merge_duplicates(store, threshold=0.80) == 1
    assert store.get_fact(a).superseded_by == b
    stale, _ = store.add_fact("User might own a bicycle", "other", 0.3)
    store._x("UPDATE facts SET updated_at=?, last_used_at=NULL WHERE id=?", (time.time() - 90 * 86400, stale))
    keep, _ = store.add_fact("User is named Sam", "person", 0.95, explicit=True)
    store._x("UPDATE facts SET updated_at=? WHERE id=?", (time.time() - 90 * 86400, keep))
    decayed, archived = decay(store, after_days=30, factor=0.5, archive_below=0.2)
    assert archived == 1 and not store.get_fact(stale).active and store.get_fact(keep).active
    sid = store.start_session(started_at=time.time() - 5 * 86400)
    store.add_turn(sid, "user", "Help me plan a trip to Japan in April")
    store.add_turn(sid, "assistant", "Sure, cherry blossom season.")
    llm = ScriptedLLM(completions=["User planned a trip to Japan in April for cherry blossoms."])
    rep = consolidate(store, llm, cfg)
    assert rep.summarized == 1
    assert "Japan" in store.session(sid)["summary"]
    assert store.search_chunks("japan trip", k=1)[0][0]["kind"] == "summary"
