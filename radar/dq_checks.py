from __future__ import annotations
import math
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)

def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute lightweight quality metrics without external deps.
    """
    n_rows, n_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    col_summaries = []
    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        info = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": miss,
            "missing_pct": (miss / n_rows * 100) if n_rows else 0.0,
            "unique": unique,
        }
        if _is_numeric(s):
            s_numeric = pd.to_numeric(s, errors="coerce")
            info.update({
                "min": float(np.nanmin(s_numeric)) if s_numeric.notna().any() else None,
                "max": float(np.nanmax(s_numeric)) if s_numeric.notna().any() else None,
                "mean": float(np.nanmean(s_numeric)) if s_numeric.notna().any() else None,
                "std": float(np.nanstd(s_numeric)) if s_numeric.notna().any() else None,
                "p95": float(np.nanpercentile(s_numeric, 95)) if s_numeric.notna().any() else None,
                "p05": float(np.nanpercentile(s_numeric, 5)) if s_numeric.notna().any() else None,
            })
            # Simple outlier flag via IQR
            q1 = np.nanpercentile(s_numeric, 25) if s_numeric.notna().any() else np.nan
            q3 = np.nanpercentile(s_numeric, 75) if s_numeric.notna().any() else np.nan
            iqr = q3 - q1 if not np.isnan(q1) and not np.isnan(q3) else np.nan
            if not np.isnan(iqr) and iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 - 1.5 * iqr
                outliers = int(((s_numeric < lower) | (s_numeric > upper)).sum())
            else:
                outliers = 0
            info["outliers_iqr"] = outliers
        elif _is_datetime(s):
            s_dt = pd.to_datetime(s, errors="coerce")
            info.update({
                "min_date": str(s_dt.min()) if s_dt.notna().any() else None,
                "max_date": str(s_dt.max()) if s_dt.notna().any() else None,
            })
        else:
            # Categorical summary
            top = s.value_counts(dropna=True).head(5)
            info["top_values"] = top.to_dict()
        col_summaries.append(info)

    result = {
        "rows": n_rows,
        "cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "columns": col_summaries,
    }
    return result

def issues_from_profile(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Turn metrics into human-friendly issues with severities.
    """
    issues = []
    n_rows = profile["rows"]
    # Duplicate rows
    if profile["duplicate_rows"] > 0:
        issues.append({
            "type": "duplicates",
            "level": "warning",
            "message": f"{profile['duplicate_rows']} duplicate rows detected",
            "suggestion": "Drop exact duplicates"
        })
    # Column specific
    for col in profile["columns"]:
        name = col["column"]
        miss = col["missing"]
        miss_pct = col["missing_pct"]
        if miss > 0:
            level = "error" if miss_pct > 20 else "warning"
            issues.append({
                "type": "missing_values",
                "column": name,
                "level": level,
                "message": f"{miss} missing values in {name} ({miss_pct:.1f} percent)",
                "suggestion": "Impute with median for numeric, mode for categorical, or flag rows"
            })
        if "outliers_iqr" in col and col["outliers_iqr"] and col["outliers_iqr"] > 0:
            issues.append({
                "type": "outliers",
                "column": name,
                "level": "info",
                "message": f"{col['outliers_iqr']} potential outliers in {name} by IQR rule",
                "suggestion": "Review distribution and cap or winsorize if needed"
            })
    return issues

def run_checks(df: pd.DataFrame) -> Dict[str, Any]:
    profile = basic_profile(df)
    issues = issues_from_profile(profile)
    return {"profile": profile, "issues": issues}

def to_mpl_missingness(df: pd.DataFrame):
    """
    Return x and y arrays for a simple missingness bar chart.
    """
    cols = list(df.columns)
    missing_counts = [int(df[c].isna().sum()) for c in cols]
    return cols, missing_counts
