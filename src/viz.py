"""Plotting helpers. Keep notebooks visually consistent."""
from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from . import config as C

sns.set_theme(style="whitegrid", context="notebook")


def plot_label_balance(df, ax=None):
    ax = ax or plt.gca()
    counts = df[C.LABEL_COL].value_counts().sort_index()
    counts.index = [C.LABEL_NAMES[i] for i in counts.index]
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Label balance")
    ax.set_ylabel("Count")
    return ax


def plot_length_distribution(df, ax=None):
    ax = ax or plt.gca()
    lens = df[C.TEXT_COL].str.split().str.len()
    sns.histplot(lens, bins=40, ax=ax)
    ax.set_title("Sentence length (words)")
    ax.set_xlabel("Words")
    return ax


def plot_confusion(y_true, y_pred, ax=None):
    ax = ax or plt.gca()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=C.LABEL_NAMES, yticklabels=C.LABEL_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return ax


def plot_kfold_metrics(df_metrics, ax=None):
    ax = ax or plt.gca()
    melt = df_metrics.melt(id_vars="fold", var_name="metric", value_name="score")
    sns.barplot(data=melt, x="metric", y="score", ax=ax)
    ax.set_title("k-fold mean metrics")
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_baseline_comparison(our_f1: float, ax=None):
    ax = ax or plt.gca()
    rows = [
        ("Spinde 2021 (distant + BABE)", 0.804),
        ("Krieger 2022 (DA-RoBERTa)", 0.814),
        ("Ours (RoBERTa-base)", our_f1),
    ]
    df = pd.DataFrame(rows, columns=["model", "f1_macro"])
    sns.barplot(data=df, x="f1_macro", y="model", ax=ax, orient="h")
    ax.set_xlim(0.6, 0.9)
    ax.set_title("Macro-F1 vs published baselines")
    return ax
