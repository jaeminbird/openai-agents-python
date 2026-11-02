from __future__ import annotations

import pytest

from agents.voice import get_sentence_based_splitter


def _stream(text: str, chunk_size: int, min_sentence_length: int = 20) -> list[str]:
    """Feed text through the splitter the way VoiceStreamedResult._add_text does."""
    split = get_sentence_based_splitter(min_sentence_length)
    buffer = ""
    spoken: list[str] = []
    for index in range(0, len(text), chunk_size):
        buffer += text[index : index + chunk_size]
        chunk, buffer = split(buffer)
        if chunk:
            spoken.append(chunk)
    if buffer.strip():
        spoken.append(buffer.strip())
    return spoken


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 5, 7, 11, 15])
@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith went to Washington. He arrived at 3 p.m. sharp.",
        "Hello there friend. How are you doing today? I am fine. Goodbye now.",
        "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.",
        "A short one. Then a much longer sentence that exceeds the minimum easily.",
    ],
    ids=["abbreviations", "questions", "many_short", "mixed_lengths"],
)
def test_streamed_text_is_spoken_without_losing_or_gluing_words(text: str, chunk_size: int) -> None:
    """Splitting must not depend on where the model's deltas happen to break.

    The buffer was stripped before splitting, which also removed the trailing space that
    separates the held-back text from the next delta. When a delta boundary landed just
    after a space, the following word was concatenated onto the previous one, so "He "
    plus "arrived" was spoken as "Hearrived".
    """
    assert " ".join(_stream(text, chunk_size)).split() == text.split()


def test_split_preserves_the_separator_before_the_next_delta() -> None:
    """The remainder must still end with the whitespace it was given."""
    split = get_sentence_based_splitter(20)

    spoken, remaining = split("This sentence is long enough to flush. He ")

    assert spoken == "This sentence is long enough to flush."
    # Without the trailing space, appending the next delta glues the words together.
    assert remaining == "He "
    assert (remaining + "arrived").split() == ["He", "arrived"]


def test_split_leaves_the_buffer_untouched_when_nothing_is_flushed() -> None:
    split = get_sentence_based_splitter(20)

    assert split("Too short. ") == ("", "Too short. ")
    assert split("   ") == ("", "   ")
    assert split("") == ("", "")


def test_split_without_trailing_whitespace_is_unchanged() -> None:
    split = get_sentence_based_splitter(20)

    assert split("This sentence is long enough to flush. He") == (
        "This sentence is long enough to flush.",
        "He",
    )


@pytest.mark.parametrize(
    "text",
    [
        "好的，这是计划。首先我们收拾行李，然后开车去机场。最后我们登机休息！你觉得怎么样？",
        "はい、これが計画です。まず荷物を"
        "まとめて、次に車で空港へ行きます。"
        "最後に飛行機に乗ります。",
    ],
    ids=["chinese", "japanese"],
)
@pytest.mark.parametrize("chunk_size", [1, 3, 7])
def test_cjk_text_is_spoken_sentence_by_sentence(text: str, chunk_size: int) -> None:
    """Full-width sentence marks have no space after them, so they must split on their own.

    Without that, a Chinese or Japanese reply was never split and text-to-speech only started
    after the whole response had streamed in.
    """
    spoken = _stream(text, chunk_size, min_sentence_length=10)

    assert len(spoken) > 1
    assert "".join(spoken) == text


def test_cjk_split_keeps_sentences_unspaced() -> None:
    split = get_sentence_based_splitter(5)

    assert split("这是第一句话。这是第二句话。还没") == (
        "这是第一句话。这是第二句话。",
        "还没",
    )
