"""BABE dataset loading, cleaning, splitting."""
from __future__ import annotations
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C


def download_babe() -> pd.DataFrame:
    """Download BABE from HuggingFace and return a normalized DataFrame.

    BABE has multiple configs/columns across versions. We normalize to (text, label).
    """
    ds = load_dataset(C.HF_DATASET)
    # Most BABE configs expose a 'train' split with 'text' and 'label' (or 'label_bias')
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split_name].to_pandas()

    # Normalize column names
    text_candidates = ["text", "sentence", "Text", "Sentence"]
    label_candidates = ["label", "label_bias", "Label", "bias"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if text_col is None or label_col is None:
        raise ValueError(f"Could not find text/label columns. Got: {df.columns.tolist()}")

    df = df.rename(columns={text_col: C.TEXT_COL, label_col: C.LABEL_COL})

    # Some BABE variants use string labels like 'Biased' / 'Non-biased'
    if df[C.LABEL_COL].dtype == object:
        mapping = {
            "Biased": 1, "biased": 1, "BIASED": 1, "1": 1, 1: 1,
            "Non-biased": 0, "non-biased": 0, "NON-BIASED": 0, "0": 0, 0: 0,
            "No agreement": None,
        }
        df[C.LABEL_COL] = df[C.LABEL_COL].map(mapping)

    df = df[[C.TEXT_COL, C.LABEL_COL]].dropna()
    df[C.LABEL_COL] = df[C.LABEL_COL].astype(int)
    df[C.TEXT_COL] = df[C.TEXT_COL].astype(str).str.strip()
    df = df[df[C.TEXT_COL].str.len() > 0].drop_duplicates(subset=[C.TEXT_COL]).reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, name: str = "babe_clean.parquet") -> str:
    out = C.DATA_PROCESSED / name
    df.to_parquet(out, index=False)
    return str(out)


def load_processed(name: str = "babe_clean.parquet") -> pd.DataFrame:
    return pd.read_parquet(C.DATA_PROCESSED / name)


def make_holdout_split(df: pd.DataFrame, test_size: float = 0.15):
    """Stratified holdout split for the final test set."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[C.LABEL_COL],
        random_state=C.SEED,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def kfold_indices(df: pd.DataFrame, n_splits: int = C.N_FOLDS):
    """Yield (fold_idx, train_idx, val_idx) for stratified k-fold CV."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.SEED)
    for i, (tr, va) in enumerate(skf.split(df[C.TEXT_COL], df[C.LABEL_COL])):
        yield i, tr, va
