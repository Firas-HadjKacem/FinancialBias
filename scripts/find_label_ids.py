import sys
from transformers import AutoTokenizer

TARGET_SENTIMENTS = ["positive", "negative", "neutral"]


def find_all_label_ids(base_tokenizer_id: str):
    """
    For a given tokenizer, find all token IDs whose decoded string
    is exactly one of: 'positive', 'negative', 'neutral'.

    Returns:
        dict: {'Positive': [...], 'Negative': [...], 'Neutral': [...]}
    """
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_id)

    hits = {t: [] for t in TARGET_SENTIMENTS}

    # Get vocab size robustly
    if hasattr(tokenizer, "vocab_size"):
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = len(tokenizer.get_vocab())

    print(f"Tokenizer: {base_tokenizer_id}")
    print(f"Vocab size: {vocab_size}")

    for tid in range(vocab_size):
        decoded = tokenizer.decode([tid])
        cleaned = decoded.strip().lower()
        if cleaned in hits:
            hits[cleaned].append(tid)

    # Build label_ids in the exact format LlamaModelWrapper expects:
    label_ids = {t.capitalize(): ids for t, ids in hits.items()}

    print("label_ids = {")
    for k, ids in label_ids.items():
        print(f"    '{k}': {ids},")
    print("}")

    return label_ids

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/find_label_ids.py <base_tokenizer_id>")
        sys.exit(1)

    base_tokenizer_id = sys.argv[1]
    find_all_label_ids(base_tokenizer_id)

if __name__ == "__main__":
    main()