from typing import Dict, Tuple

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_tokenizer_and_model(
    model_name: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
):
    """
    Load a pretrained transformer tokenizer and sequence classification model.

    Args:
        model_name: Hugging Face model checkpoint name (e.g., 'roberta-base').
        num_labels: Number of target classes.
        id2label: Mapping from integer ID -> string label.
        label2id: Mapping from string label -> integer ID.

    Returns:
        tokenizer: Tokenizer instance.
        model: AutoModelForSequenceClassification instance with correct label config.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    return tokenizer, model


def tokenize_batch(examples, tokenizer, max_length: int = 256):
    """
    Tokenize a batch of examples.

    Args:
        examples: A batch dict coming from Hugging Face datasets (contains 'text').
        tokenizer: The tokenizer returned by load_tokenizer_and_model.
        max_length: Maximum sequence length for truncation/padding.

    Returns:
        A dict with tokenized outputs (input_ids, attention_mask, etc.).
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
