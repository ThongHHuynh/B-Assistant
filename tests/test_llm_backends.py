"""Exercise the real Anthropic SDK against a local mock of the Messages streaming API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bi.brain.llm import AnthropicLLM, Done, LLMConfigError, OllamaLLM, SystemPrompt, TextDelta, ToolCall


def sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def message_events(blocks, stop="end_turn"):
    ev = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
            "model": "claude-haiku-5-5", "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 12, "output_tokens": 1}}}]
    for i, b in enumerate(blocks):
        if b["type"] == "text":
            ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
            for piece in b["pieces"]:
                ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": piece}})
        else:
            ev.append({"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": b["id"], "name": b["name"], "input": {}}})
            ev.append({"type": "content_block_delta", "index": i,
                       "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})
        ev.append({"type": "content_block_stop", "index": i})
    ev.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 9}})
    ev.append({"type": "message_stop"})
    return ev


@pytest.fixture
def mock_api():
    state = {"requests": [], "responses": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            state["requests"].append({"path": self.path, "body": body, "headers": dict(self.headers)})
            payload = sse(state["responses"].pop(0))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()


def make(cfg, url, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")
    cfg.set("llm.anthropic.base_url", url)
    return AnthropicLLM(cfg)


def test_anthropic_stream_text_and_request_shape(cfg, mock_api, monkeypatch):
    llm = make(cfg, mock_api["url"], monkeypatch)
    mock_api["responses"].append(message_events([{"type": "text", "pieces": ["Hi ", "there."]}]))
    tools = [{"name": "get_time", "description": "time", "input_schema": {"type": "object", "properties": {}}}]
    evs = list(llm.stream([{"role": "user", "content": "hello"}], tools, SystemPrompt("STATIC", "DYNAMIC")))
    assert [e.text for e in evs if isinstance(e, TextDelta)] == ["Hi ", "there."]
    done = evs[-1]
    assert isinstance(done, Done) and done.stop_reason == "end_turn" and done.text == "Hi there."
    body = mock_api["requests"][0]["body"]
    assert body["model"] == "claude-haiku-5-5" and body["stream"] is True
    assert body["system"][0] == {"type": "text", "text": "STATIC", "cache_control": {"type": "ephemeral"}}
    assert body["system"][1]["text"] == "DYNAMIC"
    assert body["output_config"] == {"effort": "low"}
    assert body["tools"][0]["eager_input_streaming"] is True
    assert "temperature" not in body


def test_anthropic_tool_call_and_replay(cfg, mock_api, monkeypatch):
    llm = make(cfg, mock_api["url"], monkeypatch)
    mock_api["responses"].append(message_events(
        [{"type": "text", "pieces": ["Checking."]},
         {"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {"tz": "UTC"}}], stop="tool_use"))
    evs = list(llm.stream([{"role": "user", "content": "time?"}], [], SystemPrompt("S")))
    calls = [e for e in evs if isinstance(e, ToolCall)]
    assert calls == [ToolCall("toolu_1", "get_time", {"tz": "UTC"})]
    done = evs[-1]
    # replay within the turn: raw blocks go back unchanged, tool results become one user message
    msgs = [{"role": "user", "content": "time?"},
            {"role": "assistant", "content": done.text, "raw": done.raw,
             "tool_calls": [{"id": "toolu_1", "name": "get_time", "input": {"tz": "UTC"}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "name": "get_time", "content": "10:00"}]
    mock_api["responses"].append(message_events([{"type": "text", "pieces": ["It's ten."]}]))
    list(llm.stream(msgs, [], SystemPrompt("S"), role="think"))
    req = mock_api["requests"][1]
    body = req["body"]
    assert body["model"] == "claude-opus-5-5" and body["output_config"] == {"effort": "medium"}
    assert body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in req["headers"].get("anthropic-beta", "")
    assert body["messages"][1]["content"][1]["type"] == "tool_use"
    assert body["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "10:00"}]}


def test_anthropic_missing_key_is_clear(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM(cfg)


def test_ollama_message_conversion():
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "get_time", "input": {}}]},
            {"role": "tool", "tool_call_id": "c1", "name": "get_time", "content": "noon"}]
    out = OllamaLLM.convert(msgs, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "get_time"
    assert out[3] == {"role": "tool", "content": "noon", "tool_name": "get_time"}


def test_ollama_unreachable_is_clear(cfg):
    cfg.set("llm.ollama.host", "http://127.0.0.1:9")
    llm = OllamaLLM(cfg)
    with pytest.raises(LLMConfigError, match="ollama serve"):
        list(llm.stream([{"role": "user", "content": "hi"}], None, SystemPrompt("S")))
