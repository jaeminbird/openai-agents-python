import re
from collections.abc import Callable

_SENTENCE_BOUNDARY = re.compile(r"((?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])\s*)")


def get_sentence_based_splitter(
    min_sentence_length: int = 20,
) -> Callable[[str], tuple[str, str]]:
    """Returns a function that splits text into chunks based on sentence boundaries.

    Args:
        min_sentence_length: The minimum length of a sentence to be included in a chunk.

    Returns:
        A function that splits text into chunks based on sentence boundaries.
    """

    def sentence_based_text_splitter(text_buffer: str) -> tuple[str, str]:
        """
        A function to split the text into chunks. This is useful if you want to split the text into
        chunks before sending it to the TTS model rather than waiting for the whole text to be
        processed.

        Args:
            text_buffer: The text to split.

        Returns:
            A tuple of the text to process and the remaining text buffer.
        """
        # Chinese and Japanese end sentences with full-width marks and no following space,
        # so those marks are boundaries on their own. Without this, CJK replies were never
        # split and text-to-speech waited for the whole response.
        parts = _SENTENCE_BOUNDARY.split(text_buffer.strip())
        sentences, separators = parts[0::2], parts[1::2]
        if len(sentences) >= 1:
            # Whitespace between sentences becomes one space; CJK sentences have none.
            combined_sentences = "".join(
                sentence + (" " if separator else "")
                for sentence, separator in zip(sentences[:-1], separators, strict=True)
            ).rstrip()
            if len(combined_sentences) >= min_sentence_length:
                # Carry any trailing whitespace over to the remainder. It separates the text
                # held back from the next streamed delta, and stripping it concatenates the
                # next word onto the last one, so "He " followed by "arrived" is spoken as
                # "Hearrived".
                trailing_whitespace = text_buffer[len(text_buffer.rstrip()) :]
                remaining_text_buffer = sentences[-1] + trailing_whitespace
                return combined_sentences, remaining_text_buffer
        return "", text_buffer

    return sentence_based_text_splitter
