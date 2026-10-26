import pytest

from bi.brain.sentences import SentenceSplitter, clean_for_speech, split_sentences


def stream(text, step=3, first=10**6):
    sp = SentenceSplitter(first_chunk_min_chars=first)
    out = []
    for i in range(0, len(text), step):
        out += sp.push(text[i:i + step])
    return out + sp.flush()


@pytest.mark.parametrize("step", [1, 2, 5, 50])
def test_basic_split_independent_of_token_size(step):
    text = "Sure thing. It's 3.5 degrees outside! Do you want a jacket? Okay."
    assert stream(text, step) == ["Sure thing.", "It's 3.5 degrees outside!", "Do you want a jacket?", "Okay."]


def test_abbreviations_initials_and_decimals():
    text = "Dr. Smith met J. R. Tolkien at 5 p.m. on Main St. yesterday. Pi is 3.14159. Done."
    assert stream(text, 4) == ["Dr. Smith met J. R. Tolkien at 5 p.m. on Main St. yesterday.", "Pi is 3.14159.", "Done."]


def test_sentence_emitted_as_soon_as_it_ends():
    sp = SentenceSplitter()
    assert sp.push("Hello there.") == []           # could still be "Hello there.com" / "3.5": wait for next char
    assert sp.push(" How") == ["Hello there."]
    assert sp.push(" are you?") == []
    assert sp.flush() == ["How are you?"]


def test_first_chunk_cut_early_at_clause():
    sp = SentenceSplitter(first_chunk_min_chars=24)
    out = sp.push("Well, according to the forecast for tomorrow, it will rain in the afternoon")
    assert out == ["Well, according to the forecast for tomorrow,"]
    out += sp.push(", so take an umbrella. Also, the wind")
    assert out[1] == "it will rain in the afternoon, so take an umbrella."   # later chunks are full sentences
    assert sp.flush() == ["Also, the wind"]


def test_markdown_is_not_read_aloud():
    text = "**Sure!** Here are options:\n\n- *Pasta* with `pesto`\n- [Salad](http://x.y)\n\n## Done"
    out = stream(text, 6)
    joined = " ".join(out)
    for junk in ("*", "`", "#", "http", "](", "- "):
        assert junk not in joined, junk
    assert "Pasta with pesto" in joined and "Salad" in joined


def test_long_run_on_is_split():
    text = "word " * 80
    out = split_sentences(text)
    assert len(out) >= 2 and all(len(s) <= 220 for s in out)


def test_clean_for_speech():
    assert clean_for_speech("1. **Bold** and _it_ & more") == "Bold and it and more"
    assert clean_for_speech("```python\nprint(1)\n``` after") == "after"
