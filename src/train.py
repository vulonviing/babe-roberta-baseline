"""Training wrapper around HuggingFace Trainer."""
from __future__ import annotations
import os
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

from . import config as C
from .model import get_tokenizer, get_model, tokenize_dataframe


def train_one(
    train_df,
    val_df,
    run_name: str,
    output_subdir: str,
    epochs: int = C.NUM_EPOCHS,
    use_wandb: bool = C.WANDB_ENABLED,
):
    """Fine-tune RoBERTa-base on (train_df, val_df). Returns (trainer, metrics)."""
    from .evaluate import compute_metrics  # lazy import to avoid circular dep
    set_seed(C.SEED)
    tokenizer = get_tokenizer()
    model = get_model(num_labels=2)

    train_ds = tokenize_dataframe(train_df, tokenizer)
    val_ds = tokenize_dataframe(val_df, tokenizer)

    output_dir = str(C.MODELS_DIR / output_subdir)
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=C.BATCH_SIZE,
        per_device_eval_batch_size=C.EVAL_BATCH_SIZE,
        learning_rate=C.LEARNING_RATE,
        weight_decay=C.WEIGHT_DECAY,
        warmup_ratio=C.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=1,
        seed=C.SEED,
        report_to=("wandb" if use_wandb else "none"),
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    return trainer, metrics
