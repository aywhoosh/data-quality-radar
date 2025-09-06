from __future__ import annotations
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

def _mode(series: pd.Series):
    try:
        return series.mode(dropna=True).iloc[0]
    except Exception:
        return None

def auto_repair(df: pd.DataFrame, strategy: str = "safe") -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Reversible, conservative repairs.
    - Drop exact duplicate rows
    - Fill numeric NaNs with median
    - Fill categorical NaNs with mode
    Returns cleaned_df and a changelog list of operations.
    """
    cleaned = df.copy(deep=True)
    changelog: List[Dict[str, Any]] = []

    # 1) Drop duplicates
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(keep="first")
    dropped = before - len(cleaned)
    if dropped > 0:
        changelog.append({
            "op": "drop_duplicates",
            "rows_removed": int(dropped)
        })

    # 2) Impute per column
    for col in cleaned.columns:
        s = cleaned[col]
        miss = int(s.isna().sum())
        if miss == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            fill_value = float(s.median())
            cleaned[col] = s.fillna(fill_value)
            changelog.append({
                "op": "impute_median",
                "column": col,
                "missing_filled": miss,
                "value": fill_value
            })
        else:
            m = _mode(s)
            if m is not None:
                cleaned[col] = s.fillna(m)
                changelog.append({
                    "op": "impute_mode",
                    "column": col,
                    "missing_filled": miss,
                    "value": str(m)
                })
            else:
                # If no mode, leave as is
                changelog.append({
                    "op": "impute_mode_skipped",
                    "column": col,
                    "missing_unfilled": miss
                })
    return cleaned, changelog
