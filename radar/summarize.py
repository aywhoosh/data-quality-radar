from __future__ import annotations
from typing import Dict, Any, List
import math

def narrate(report: Dict[str, Any], changelog: List[Dict[str, Any]]) -> str:
    profile = report["profile"]
    issues = report["issues"]
    n_rows = profile["rows"]
    n_cols = profile["cols"]
    dup = profile["duplicate_rows"]
    missing_total = sum(c["missing"] for c in profile["columns"])
    # Build sentences
    lines = []
    lines.append(f"The file has {n_rows} rows and {n_cols} columns, with {dup} duplicate rows and {missing_total} total missing cells.")
    # Top two missing columns
    top_missing = sorted(profile["columns"], key=lambda c: c["missing"], reverse=True)[:2]
    for c in top_missing:
        if c["missing"] > 0:
            lines.append(f"Column {c['column']} has {c['missing']} missing values which is {c['missing_pct']:.1f} percent.")
            break
    # Outliers note
    outs = [c for c in profile["columns"] if "outliers_iqr" in c and c["outliers_iqr"] and c["outliers_iqr"] > 0]
    if outs:
        names = ", ".join(o["column"] for o in outs[:3])
        lines.append(f"Potential outliers identified by IQR in: {names}.")
    # Repairs
    if changelog:
        ops = [e["op"] for e in changelog]
        lines.append(f"Applied repairs: {', '.join(sorted(set(ops)))}.")
        drops = next((e for e in changelog if e['op'] == 'drop_duplicates'), None)
        if drops:
            lines.append(f"Removed {drops['rows_removed']} duplicate rows.")
        impute_notes = [e for e in changelog if e['op'].startswith('impute_')]
        if impute_notes:
            total_imputed = sum(e.get('missing_filled', 0) for e in impute_notes)
            lines.append(f"Filled {total_imputed} missing values using median for numeric and mode for categorical columns.")
    else:
        lines.append("No automatic repairs were applied.")
    # Next steps
    lines.append("Next steps: validate business rules for key columns, review outliers, and consider stricter expectations for future uploads.")
    return " ".join(lines)
