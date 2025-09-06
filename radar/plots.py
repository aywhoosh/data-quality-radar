from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(ax, series: pd.Series, bins: int = 30, title: str = ""):
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    ax.hist(s, bins=bins)
    ax.set_title(title or "Histogram")
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")

def plot_bar_counts(ax, series: pd.Series, top_n: int = 15, title: str = ""):
    counts = series.astype("string").value_counts().head(top_n)
    ax.bar(np.arange(len(counts)), counts.values)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index.tolist(), rotation=45, ha="right")
    ax.set_title(title or "Counts")
    ax.set_ylabel("count")

def plot_group_mean(ax, df: pd.DataFrame, cat: str, num: str, title: str = ""):
    tmp = df[[cat, num]].copy()
    tmp[num] = pd.to_numeric(tmp[num], errors="coerce")
    means = tmp.groupby(cat)[num].mean().dropna()
    ax.bar(np.arange(len(means)), means.values)
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(means.index.tolist(), rotation=45, ha="right")
    ax.set_title(title or f"Mean of {num} by {cat}")
    ax.set_ylabel(f"mean({num})")

def plot_scatter_colored(ax, df: pd.DataFrame, x: str, y: str, color_cat: str | None = None, alpha: float = 0.7, title: str = ""):
    xs = pd.to_numeric(df[x], errors="coerce")
    ys = pd.to_numeric(df[y], errors="coerce")
    mask = xs.notna() & ys.notna()
    xs, ys = xs[mask], ys[mask]
    if color_cat and color_cat in df.columns:
        cats = df.loc[mask, color_cat].astype("category")
        codes = cats.cat.codes
        sc = ax.scatter(xs, ys, c=codes, alpha=alpha)
        ax.set_title(title or f"{x} vs {y} colored by {color_cat}")
    else:
        ax.scatter(xs, ys, alpha=alpha)
        ax.set_title(title or f"{x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)

def plot_corr_heatmap(ax, df: pd.DataFrame, title: str = "Correlation heatmap"):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")
        return
    corr = df[num_cols].corr()
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(np.arange(len(num_cols)))
    ax.set_yticks(np.arange(len(num_cols)))
    ax.set_xticklabels(num_cols, rotation=45, ha="right")
    ax.set_yticklabels(num_cols)
    ax.set_title(title)
    # annotate
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
