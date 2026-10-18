"""
BPE Tokenizer starter
"""

# pylint:disable=too-many-locals, unused-variable, import-error


from main import (  # type: ignore
    collect_frequencies,
    decode,
    encode,
    get_vocabulary,
    load_vocabulary,
    train,
)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def _read_encoded(path: str) -> list[int]:
    return [int(item) for item in _read_text(path).split()]

def main() -> None:
    """
    Launches an implementation
    """

    base = "lab_2_tokenize_by_bpe/assets"

    text = _read_text(f"{base}/text.txt")
    word_frequencies = collect_frequencies(
        text, start_of_word=None, end_of_word="</s>"
    )
    print("== Шаг 2. Частотный словарь (первые 5 записей) ===")
    for index, (word, freq) in enumerate(word_frequencies.items()):
        if index >= 5:
            break
        print(f" {word}: {freq}")
    print(f" ... всего уникальных слов: {len(word_frequencies)}")

    trained = train(word_frequencies, 100)
    print("\n== Шаг 5. Токенизатор обучен (100 слияний) ==")
    print(f"  Записей в частотном словаре после обучения: {len(trained)}")
    print("  Первые 10 токенов после обучения: ")
    for word in list(trained.keys())[:10]:
        print(f"    {word}")

    vocab = get_vocabulary(trained, "<unk>")
    print("\n== Шаг 6. Словарь токенов ==")
    for token in list(vocab.keys())[:10]:
        print(f"  {token!r}: {vocab[token]}")

    sample_word = next(iter(trained))
    sample_ids = [vocab[token] for token in sample_word]
    decoded = decode(sample_ids, vocab, "</s>")
    print("\n== Шаг 7. Демонстрация декодирования ==")
    print(f"  Слово из обученного словаря: {sample_word}")
    print(f"  Его идентификаторы: {sample_ids}")
    print(f"  Декодировано: {decoded!r}")

    real_vocab = load_vocabulary(f"{base}/vocab.json")
    ru_raw = _read_text(f"{base}/ru_raw.txt")
    encoded = encode(
        ru_raw,
        real_vocab,
        start_of_word_token="\u2581",
        end_of_word_token=None,
        unknown_token="<unk>",
    )
    reference = _read_encoded(f"{base}/ru_encoded.txt")
    print("\n== Шаг 10. Кодирование ru_raw.txt ==")
    if encoded is None:
        print(" encode вернул None - проверь vocab.json / unknown_token")
    else:
        print(f"  Первые 20 токенов: {encoded[:20]}")
        print(f"  Эталон (первые 20): {reference[:20]}")
        print(f"  Совпадение первых 20 токенов с эталоном: {encoded[:20] == reference[:20]}")


    if __name__ == "__main__":
    main()
