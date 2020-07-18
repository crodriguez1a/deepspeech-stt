from typing import List, Sequence, Any

from nltk.corpus import words

english_lexicon: list = words.words()


def english_scoring(sentences: List[str]) -> List[float]:
    """
    Calculate the percentage of english words in each sentence
    """
    english_word_counts: list = []
    for text in sentences:
        wrds: list = text.split()
        word_count: int = len(wrds)
        if word_count:
            english_words: list = [w for w in wrds if w in english_lexicon]
            english_word_counts.append(len(english_words) / word_count)

    return english_word_counts or [0.0] * len(sentences)
