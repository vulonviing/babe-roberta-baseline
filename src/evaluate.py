"""Metrics + k-fold cross-validation orchestration."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from . import config as C
from .data import kfold_indices
from .train import train_one


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_biased": f1_score(labels, preds, average="binary", pos_label=1),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def run_kfold_cv(df, n_folds: int = C.N_FOLDS, epochs: int = C.NUM_EPOCHS):
    """Run stratified k-fold CV. Returns a DataFrame of fold metrics."""
    rows = []
    for fold_idx, tr_idx, va_idx in kfold_indices(df, n_splits=n_folds):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[va_idx].reset_index(drop=True)
        _, metrics = train_one(
            train_df,
            val_df,
            run_name=f"fold-{fold_idx}",
            output_subdir=f"fold_{fold_idx}",
            epochs=epochs,
        )
        metrics["fold"] = fold_idx
        rows.append(metrics)

    df_metrics = pd.DataFrame(rows)
    out_csv = C.RESULTS_DIR / "kfold_metrics.csv"
    df_metrics.to_csv(out_csv, index=False)

    summary = {
        "mean": df_metrics.drop(columns=["fold"]).mean().to_dict(),
        "std": df_metrics.drop(columns=["fold"]).std().to_dict(),
    }
    with open(C.RESULTS_DIR / "kfold_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return df_metrics, summary


def predict_dataframe(trainer, df):
    """Use a trained Trainer to predict labels for a DataFrame."""
    from .model import tokenize_dataframe
    tok = getattr(trainer, "processing_class", None) or trainer.tokenizer
    ds = tokenize_dataframe(df, tok)
    out = trainer.predict(ds)
    preds = np.argmax(out.predictions, axis=-1)
    return preds, out.label_ids


def error_analysis(df, preds, labels, n: int = 20):
    """Return up to n misclassified examples."""
    df = df.copy().reset_index(drop=True)
    df["pred"] = preds
    df["label"] = labels
    wrong = df[df["pred"] != df["label"]].head(n)
    return wrong
