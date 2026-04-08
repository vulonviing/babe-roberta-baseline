"""Tokenizer + model factory."""
from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from . import config as C


def get_tokenizer():
    return AutoTokenizer.from_pretrained(C.MODEL_NAME)


def get_model(num_labels: int = 2):
    return AutoModelForSequenceClassification.from_pretrained(
        C.MODEL_NAME,
        num_labels=num_labels,
        id2label={i: n for i, n in enumerate(C.LABEL_NAMES)},
        label2id={n: i for i, n in enumerate(C.LABEL_NAMES)},
    )


def tokenize_dataframe(df, tokenizer):
    """Return a HuggingFace Dataset with input_ids/attention_mask/labels."""
    from datasets import Dataset
    ds = Dataset.from_pandas(df[[C.TEXT_COL, C.LABEL_COL]].reset_index(drop=True))

    def _tok(batch):
        enc = tokenizer(
            batch[C.TEXT_COL],
            truncation=True,
            padding=False,
            max_length=C.MAX_LENGTH,
        )
        enc["labels"] = batch[C.LABEL_COL]
        return enc

    ds = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    return ds
