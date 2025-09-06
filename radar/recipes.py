from __future__ import annotations
from typing import List, Dict, Any, Sequence, Tuple
import numpy as np
import pandas as pd

def impute_mode(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    fixed = df.copy(deep=True)
    log: List[Dict[str, Any]] = []
    for col in cols:
        if col not in fixed.columns:
            continue
        miss = int(fixed[col].isna().sum())
        if miss == 0:
            continue
        mode_val = fixed[col].mode(dropna=True)
        if not mode_val.empty:
            fill_value = mode_val.iloc[0]
            fixed[col] = fixed[col].fillna(fill_value)
            log.append({"op": "impute_mode", "column": col, "missing_filled": miss, "value": str(fill_value)})
    return fixed, log

def impute_group_median(df: pd.DataFrame, target: str, by: Sequence[str]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    fixed = df.copy(deep=True)
    log: List[Dict[str, Any]] = []
    if target not in fixed.columns:
        return fixed, log
    if not pd.api.types.is_numeric_dtype(fixed[target]):
        # Try coercion
        fixed[target] = pd.to_numeric(fixed[target], errors="coerce")
    before_missing = int(fixed[target].isna().sum())
    if before_missing == 0 or len(by) == 0:
        return fixed, log
    # Compute group medians
    med = fixed.groupby(list(by))[target].median()
    def fill_row(row):
        if pd.isna(row[target]):
            key = tuple(row[b] for b in by)
            try:
                return med.loc[key]
            except KeyError:
                return row[target]
        return row[target]
    fixed[target] = fixed.apply(fill_row, axis=1)
    after_missing = int(fixed[target].isna().sum())
    filled = before_missing - after_missing
    if filled > 0:
        log.append({"op": "impute_group_median", "column": target, "by": list(by), "missing_filled": int(filled)})
    return fixed, log

def add_known_indicator(df: pd.DataFrame, col: str, name: str | None = None, drop_original: bool = False):
    fixed = df.copy(deep=True)
    if col not in fixed.columns:
        return fixed, []
    ind_name = name or f"{col}Known"
    fixed[ind_name] = fixed[col].notna().astype("int64")
    log = [{"op": "add_indicator", "source": col, "indicator": ind_name}]
    if drop_original:
        fixed = fixed.drop(columns=[col])
        log.append({"op": "drop_column", "column": col})
    return fixed, log

def iqr_bounds(series: pd.Series, factor: float = 1.5):
    s = pd.to_numeric(series, errors="coerce")
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return lower, upper, iqr

def winsorize_iqr(df: pd.DataFrame, cols: Sequence[str], factor: float = 1.5, suffix: str = "_w"):
    fixed = df.copy(deep=True)
    log: List[Dict[str, Any]] = []
    for col in cols:
        if col not in fixed.columns:
            continue
        if not pd.api.types.is_numeric_dtype(fixed[col]):
            fixed[col] = pd.to_numeric(fixed[col], errors="coerce")
        lower, upper, iqr = iqr_bounds(fixed[col], factor=factor)
        before_out = int(((fixed[col] < lower) | (fixed[col] > upper)).sum())
        new_col = f"{col}{suffix}"
        fixed[new_col] = fixed[col].clip(lower=lower, upper=upper)
        log.append({
            "op": "winsorize_iqr",
            "column": col,
            "lower": float(lower) if pd.notna(lower) else None,
            "upper": float(upper) if pd.notna(upper) else None,
            "iqr": float(iqr) if pd.notna(iqr) else None,
            "outliers_capped": before_out,
            "new_column": new_col
        })
    return fixed, log
