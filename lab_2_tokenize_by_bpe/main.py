"""
Lab 2.

BPE and machine translation evaluation
"""

import json
from typing import Sequence, cast

# pylint:disable=unused-argument

def prepare_word(
    raw_word: str, start_of_word: str | None, end_of_word: str | None
) -> tuple[str, ...] | None:
    """
    Tokenize a word into characters and attach optional boundary tokens.

    Args:
        raw_word (str): Original word
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str | None): A token that signifies the end of word

    Returns:
        tuple[str, ...] | None: Preprocessed word

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(raw_word, str):
        return None
    if start_of_word is not None and not isinstance(start_of_word, str):
        return None
    if end_of_word is not None and not isinstance(end_of_word, str):
        return None
    tokens = []
    if start_of_word is not None:
        tokens.append(start_of_word)
    tokens.extend(raw_word)
    if end_of_word is not None:
        tokens.append(end_of_word)
    return tuple(tokens)

def collect_frequencies(
    text: str, start_of_word: str | None, end_of_word: str
) -> dict[tuple[str, ...], int] | None:
    """
    Count number of occurrences of each word.

    Args:
        text (str): Original text with no preprocessing
        start_of_word (str | None): A token that signifies the start of word
        end_of_word (str): A token that signifies the end of word

    Returns:
        dict[tuple[str, ...], int] | None: Dictionary where key - preprocessed word,
            value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """

    if not isinstance(text, str):
        return None
    frequencies = {}
    for raw_word in text.split():
        prepared = prepare_word(raw_word, start_of_word, end_of_word)
        if prepared is None:
            return None
        frequencies[prepared] = frequencies.get(prepared, 0) + 1
    return frequencies

def count_tokens_pairs(
    word_frequencies: dict[tuple[str, ...], int],
) -> dict[tuple[str, str], int] | None:
    """
    Count number of occurrences of each pair of subsequent tokens.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences

    Returns:
        dict[tuple[str, str], int] | None: A dictionary where
            key - token pair, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(word_frequencies, dict):
        return None
    pair_counts = {}
    for word, word_count in word_frequencies.items():
        for index in range(len(word) - 1):
            pair = (word[index], word[index + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + word_count
    return pair_counts

def merge_tokens(
    word_frequencies: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int] | None:
    """
    Update word frequency dictionary by replacing a pair of tokens with a merged one.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        pair (tuple[str, str]): A pair of tokens to be merged

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(word_frequencies, dict):
        return None
    if (
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not all(isinstance(token, str) for token in pair)
    ):
        return None
    first, second = pair
    merged_token = first + second
    new_frequencies = {}
    for word, word_count in word_frequencies.items():
        new_word = []
        index = 0
        while index < len(word):
            if (
                index < len(word) - 1
                and word[index] == first
                and word[index + 1] == second
            ):
                new_word.append(merged_token)
                index += 2
            else:
                new_word.append(word[index])
                index += 1
        new_key = tuple(new_word)
        new_frequencies[new_key] = new_frequencies.get(new_key, 0) + word_count
    return new_frequencies

def train(
    word_frequencies: dict[tuple[str, ...], int] | None, num_merges: int
) -> dict[tuple[str, ...], int] | None:
    """
    Create required number of new tokens by merging existing ones.

    Args:
        word_frequencies (dict[tuple[str, ...], int] | None): A dictionary where
            key - preprocessed word, value - number of occurrences
        num_merges (int): Required number of new tokens

    Returns:
        dict[tuple[str, ...], int] | None: A dictionary where
            key - preprocessed word, value - number of occurrences

    In case of corrupt input arguments or functions used return None,
    None is returned
    """

    if not isinstance(word_frequencies, dict):
        return None
    if not isinstance(num_merges, int) or num_merges < 0:
        return None
    current = word_frequencies
    for _ in range(num_merges):
        pair_counts = count_tokens_pairs(current)
        if not pair_counts:
            break
        best_pair = None
        best_score = None
        for p, c in pair_counts.items():
            merged = p[0] + p[1]
            score = (-c, -len(merged), merged)
            if best_pair is None or score < best_score:
                best_score = score
                best_pair = p
        if best_pair is not None:
            next_current = merge_tokens(current, best_pair)
            if next_current is None:
                return None
            current = next_current
    return current

def get_vocabulary(
    word_frequencies: dict[tuple[str, ...], int], unknown_token: str
) -> dict[str, int] | None:
    """
    Establish correspondence between tokens and its integer identifier.

    Args:
        word_frequencies (dict[tuple[str, ...], int]): A dictionary where
            key - preprocessed word, value - number of occurrences
        unknown_token (str): A token to signify an unknown token

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(word_frequencies, dict):
        return None
    if not isinstance(unknown_token, str):
        return None
    unique_tokens = set()
    for word in word_frequencies:
        for token in word:
            if token == "":
                continue
            unique_tokens.add(token)
            for char in token:
                unique_tokens.add(char)
    unique_tokens.add(unknown_token)
    sorted_tokens = sorted(unique_tokens, key=lambda token: (-len(token), token))
    return {token: index for index, token in enumerate(sorted_tokens)}

def decode(
    encoded_text: Sequence[int] | None,
    vocabulary: dict[str, int] | None,
    end_of_word_token: str | None,
) -> str | None:
    """
    Translate encoded sequence into decoded one.

    Args:
        encoded_text (Sequence[int] | None): A sequence of token identifiers
        vocabulary (dict[str, int] | None): A dictionary where
            key - token, value - identifier
        end_of_word_token (str | None): An end-of-word token

    Returns:
        str | None: Decoded sequence

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(encoded_text, (list, tuple)) or not encoded_text:
        return None
    if not isinstance(vocabulary, dict):
        return None
    if end_of_word_token is not None and not isinstance(end_of_word_token, str):
        return None
    id_to_token = {token_id: token for token, token_id in vocabulary.items()}
    pieces = []
    for token_id in encoded_text:
        token = id_to_token.get(token_id)
        if token is None:
            return None
        pieces.append(token)
    text = "".join(pieces)
    if end_of_word_token is not None:
        text = text.replace(end_of_word_token, " ")
    return text

def tokenize_word(
    word: tuple[str, ...], vocabulary: dict[str, int], end_of_word: str | None, unknown_token: str
) -> list[int] | None:
    """
    Split word into tokens.

    Args:
        word (tuple[str, ...]): Preprocessed word
        vocabulary (dict[str, int]): A dictionary where key - token, value - identifier
        end_of_word (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments, None is returned
    """

    if (
        not isinstance(word, (tuple, list))
        or not isinstance(vocabulary, dict)
        or not isinstance(unknown_token, str)
    ):
        return None
    if unknown_token not in vocabulary:
        return None
    parts = list(word)
    candidates = sorted(
        (token for token in vocabulary if token != unknown_token),
        key=lambda token: (-len(token), token),
    )
    for candidate in candidates:
        if not candidate:
            continue
        candidate_chars = list(candidate)
        token_len = len(candidate)
        index = 0
        while index < len(parts):
            if parts[index:index + token_len] == candidate_chars:
                parts[index:index + token_len] = [candidate]
            index += 1
    unknown_id = vocabulary[unknown_token]
    token_ids = []
    for part in parts:
        token_ids.append(vocabulary.get(part, unknown_id))
    return token_ids

def load_vocabulary(vocab_path: str) -> dict[str, int] | None:
    """
    Read and retrieve dictionary of type <token: identifier>.

    Args:
        vocab_path (str): A path to the saved vocabulary

    Returns:
        dict[str, int] | None: A dictionary where key - token, value - identifier

    In case of corrupt input arguments, None is returned
    """

    if not isinstance(vocab_path, str):
        return None
    with open(vocab_path, "r", encoding="utf-8") as file:
        return cast(dict[str, int], json.load(file))

def encode(
    original_text: str,
    vocabulary: dict[str, int] | None,
    start_of_word_token: str | None,
    end_of_word_token: str | None,
    unknown_token: str,
) -> list[int] | None:
    """
    Translate original text into a sequence of token identifiers.

    Args:
        original_text (str): Original text
        vocabulary (dict[str, int] | None): A dictionary where key - token, value - identifier
        start_of_word_token (str | None): A start-of-word token
        end_of_word_token (str | None): An end-of-word token
        unknown_token (str): A token that signifies unknown sequence

    Returns:
        list[int] | None: A list of token identifiers

    In case of corrupt input arguments or functions used return None,
    None is returned
    """

    if original_text is None or vocabulary is None:
        return None
    encoded: list[int] = []
    for raw_word in original_text.split():
        prepared = prepare_word(raw_word, start_of_word_token, end_of_word_token)
        if prepared is None:
            return None
        word_ids = tokenize_word(
            prepared, vocabulary, end_of_word_token, unknown_token
        )
        if word_ids is None:
            return None
        encoded.extend(word_ids)
    return encoded

def collect_ngrams(text: str, order: int) -> list[tuple[str, ...]] | None:
    """
    Extract n-grams from the given sequence.

    Args:
        text (str): Original text
        order (int): Required number of elements in a single n-gram

    Returns:
        list[tuple[str, ...]] | None: A sequence of n-grams

    In case of corrupt input arguments, None is returned
    """


def calculate_precision(
    actual: Sequence[tuple[str, ...]], reference: Sequence[tuple[str, ...]]
) -> float | None:
    """
    Compare two sequences by virtue of Precision metric.

    Args:
        actual (Sequence[tuple[str, ...]]): Predicted sequence of n-grams
        reference (Sequence[tuple[str, ...]]): Expected sequence of n-grams

    Returns:
        float | None: Value of Precision metric

    In case of corrupt input arguments, None is returned.
    """


def calculate_geo_mean(precisions: Sequence[float], max_order: int) -> float | None:
    """
    Compute geometric mean of sequence of values.

    Args:
        precisions (Sequence[float]): A sequence of Precision values
        max_order (int): Maximum length of n-gram considered

    Returns:
        float | None: A value of geometric mean of Precision metric

    In case of corrupt input arguments, None is returned
    """


def calculate_bleu(actual: str | None, reference: str, max_order: int = 3) -> float | None:
    """
    Compare two sequences by virtue of BLEU metric.

    Args:
        actual (str | None): Predicted sequence
        reference (str): Expected sequence
        max_order (int, optional): Max length of n-gram to consider for comparison. Defaults to 3

    Returns:
        float | None: A value of BLEU metric

    In case of corrupt input arguments or functions used return None,
    None is returned
    """
