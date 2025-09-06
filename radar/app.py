import io
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from core.io import load_csv
from radar.dq_checks import run_checks, to_mpl_missingness
from radar.repair import auto_repair
from radar.summarize import narrate
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


st.set_page_config(page_title="Data Quality Radar", layout="wide")
st.title("Data Quality Radar")
st.caption("Upload a CSV. Get a quality report, reversible fixes, and a narrative summary.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is None:
    st.info("Waiting for a CSV upload to begin analysis.")
    st.stop()

with st.spinner("Reading CSV..."):
    df = load_csv(uploaded)


# --------- Minimal styling ---------
st.markdown("""
    <style>
    .smallcaps { letter-spacing: .5px; text-transform: uppercase; font-size: 0.8rem; color: #666; }
    .tight { margin-top: -12px; }
    .stMetric { background: rgba(0,0,0,0.02); padding: 0.6rem; border-radius: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

tab_overview, tab_eda, tab_recipes, tab_downloads = st.tabs(["Overview", "EDA", "Recipes", "Downloads"])

with tab_overview:
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    # Run checks
    with st.spinner("Running checks..."):
        report = run_checks(df)

    profile = report["profile"]
    issues = report["issues"]

    # Metrics row
    cols_m = st.columns(3)
    cols_m[0].metric("Rows", profile["rows"])
    cols_m[1].metric("Columns", profile["cols"])
    cols_m[2].metric("Duplicate rows", profile["duplicate_rows"])

    st.markdown('<span class="smallcaps">Missingness</span>', unsafe_allow_html=True)
    cols, miss = to_mpl_missingness(df)
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.4), 3.5))
    ax.bar(range(len(cols)), miss)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_ylabel("Missing count")
    ax.set_title("Missing values per column")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Issues")
    if not issues:
        st.success("No issues detected by basic checks.")
    else:
        for itm in issues:
            level = itm.get("level", "info")
            msg = itm["message"]
            sug = itm.get("suggestion", "")
            if level == "error":
                st.error(f"{msg} Suggestion: {sug}")
            elif level == "warning":
                st.warning(f"{msg} Suggestion: {sug}")
            else:
                st.info(f"{msg} Suggestion: {sug}")

with tab_eda:
    from radar.plots import plot_hist, plot_bar_counts, plot_corr_heatmap, plot_group_mean, plot_scatter_colored

    st.subheader("Quick EDA")
    ncols = df.select_dtypes(include=[np.number]).columns.tolist()
    ccols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Univariate
    st.markdown("**Univariate**")
    col1, col2 = st.columns(2)
    with col1:
        num_choice = st.selectbox("Numeric column", ncols or ["(none)"], index=0 if ncols else None)
        if ncols:
            fig1, ax1 = plt.subplots(figsize=(5.5, 3.5))
            plot_hist(ax1, df[num_choice], title=f"Histogram: {num_choice}")
            st.pyplot(fig1, clear_figure=True)
    with col2:
        cat_choice = st.selectbox("Categorical column", ccols or ["(none)"], index=0 if ccols else None)
        if ccols:
            fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
            plot_bar_counts(ax2, df[cat_choice], title=f"Counts: {cat_choice}")
            st.pyplot(fig2, clear_figure=True)

    # Bivariate
    st.markdown("**Bivariate**")
    col3, col4 = st.columns(2)
    with col3:
        if ccols and ncols:
            cat_bi = st.selectbox("Category for mean comparison", ccols, key="eda_cat_bi")
            num_bi = st.selectbox("Numeric for mean comparison", ncols, key="eda_num_bi")
            fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
            plot_group_mean(ax3, df, cat_bi, num_bi, title=f"Mean of {num_bi} by {cat_bi}")
            st.pyplot(fig3, clear_figure=True)
    with col4:
        if len(ncols) >= 2:
            x_sc = st.selectbox("X numeric", ncols, key="eda_x_sc")
            y_sc = st.selectbox("Y numeric", [c for c in ncols if c != x_sc], key="eda_y_sc")
            color_cat = st.selectbox("Color by category (optional)", ["(none)"] + ccols, key="eda_color")
            fig4, ax4 = plt.subplots(figsize=(5.5, 3.5))
            plot_scatter_colored(ax4, df, x_sc, y_sc, None if color_cat == "(none)" else color_cat, alpha=0.7)
            st.pyplot(fig4, clear_figure=True)

    # Correlation
    st.markdown("**Correlation**")
    fig5, ax5 = plt.subplots(figsize=(6.5, 4.5))
    plot_corr_heatmap(ax5, df)
    st.pyplot(fig5, clear_figure=True)

with tab_recipes:
    st.subheader("Notebook-inspired recipes")
    st.caption("These are optional. They mirror common steps like mode imputation, group median imputation, presence indicators, and IQR winsorization.")
    from radar.recipes import impute_mode, impute_group_median, add_known_indicator, winsorize_iqr

    work = df.copy()
    change_log = []

    st.markdown("**1) Mode impute for categoricals**")
    cat_opts = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_select = st.multiselect("Columns to fill with mode", cat_opts, default=[c for c in cat_opts if df[c].isna().sum() > 0][:2])
    if st.button("Apply mode impute", key="btn_mode"):
        work, log = impute_mode(work, cat_select)
        change_log.extend(log)
        st.success("Mode impute applied.")

    st.markdown("**2) Group median impute for a numeric target**")
    num_opts = df.select_dtypes(include=[np.number]).columns.tolist()
    target = st.selectbox("Target numeric column", num_opts or ["(none)"], index=0 if num_opts else None)
    group_by = st.multiselect("Group by columns", cat_opts, max_selections=2)
    if st.button("Apply group median impute", key="btn_gmed"):
        work, log = impute_group_median(work, target, group_by)
        change_log.extend(log)
        st.success("Group median impute applied.")

    st.markdown("**3) Presence indicator for a sparse column**")
    ind_col = st.selectbox("Column to create indicator for", df.columns, index=min( len(df.columns)-1, max(0, list(df.columns).index(next((c for c in df.columns if df[c].isna().sum()>0), df.columns[0])) ) ))
    drop_src = st.checkbox("Drop original column after indicator is created", value=False)
    if st.button("Add indicator", key="btn_ind"):
        work, log = add_known_indicator(work, ind_col, None, drop_original=drop_src)
        change_log.extend(log)
        st.success("Indicator added.")

    st.markdown("**4) Winsorize numeric columns by IQR (cap extremes)**")
    nums_for_win = st.multiselect("Numeric columns to winsorize", num_opts, default=num_opts[:1])
    factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
    if st.button("Apply winsorization", key="btn_win"):
        work, log = winsorize_iqr(work, nums_for_win, factor=factor, suffix="_w")
        change_log.extend(log)
        st.success("Winsorization applied.")

    st.markdown("**Preview after recipes**")
    st.dataframe(work.head(20), use_container_width=True)

    # Save results in session state to be used in Downloads tab
    st.session_state['recipe_df'] = work
    st.session_state['recipe_log'] = change_log

with tab_downloads:
    st.subheader("Downloads")
    # Narrative from the original report
    from radar.summarize import narrate
    report = run_checks(df)
    summary = narrate(report, [])

    # Which data to download: original auto repair vs recipe output vs raw
    mode = st.radio("Choose dataset to download", ["Original upload", "Auto-repaired (median/mode + drop duplicates)", "Recipes output (from previous tab)"], index=1)
    if mode == "Original upload":
        to_save = df.copy()
    elif mode == "Auto-repaired (median/mode + drop duplicates)":
        from radar.repair import auto_repair
        to_save, _log = auto_repair(df)
    else:
        to_save = st.session_state.get('recipe_df', df)

    cleaned_csv = to_save.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", cleaned_csv, file_name="export.csv", mime="text/csv")

    summary_txt = summary.encode("utf-8")
    st.download_button("Download summary text", summary_txt, file_name="summary.txt", mime="text/plain")

    # If user used recipes, offer log
    rlog = st.session_state.get('recipe_log', [])
    if rlog:
        rlog_json = json.dumps(rlog, indent=2).encode("utf-8")
        st.download_button("Download recipes changelog", rlog_json, file_name="recipes_changelog.json", mime="application/json")
