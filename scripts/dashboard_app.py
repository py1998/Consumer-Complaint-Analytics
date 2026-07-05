"""
Stage 8 -- Consumer Complaint Analytics Dashboard (Streamlit).

WHAT THIS IS
  A plain-Python interactive dashboard over the four-category CFPB complaint subset.


RUN
  streamlit run scripts/dashboard_app.py
  -> opens http://localhost:8501
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------- config / constants
DATA_DIR = Path("data/processed")
STRUCTURED_FILE = DATA_DIR / "dash_structured.parquet"
CUBE_FILE = DATA_DIR / "dash_templated_agg.parquet"

CATEGORIES = ["Credit reporting", "Debt collection", "Mortgage", "Credit card"]
BUREAUS = ["TRANSUNION INTERMEDIATE HOLDINGS, INC.", "EQUIFAX, INC.", "Experian Information Solutions Inc."]

# 3-level relief grouping (Stage 6). Everything not listed is "Other / in progress" (excluded from
# the 3-level analysis, shown separately).
RELIEF_MAP = {
    "Closed with monetary relief": "Monetary relief",
    "Closed with non-monetary relief": "Non-monetary relief",
    "Closed with explanation": "No relief",
    "Closed without relief": "No relief",
    "Closed": "No relief",
}
RELIEF_ORDER = ["Monetary relief", "Non-monetary relief", "No relief", "Other / in progress"]

st.set_page_config(page_title="Consumer Complaint Analytics", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------- cached data loaders
@st.cache_data(show_spinner="Loading structured data (once) ...")
def load_structured() -> pd.DataFrame:
    # Columns come back as pandas 'category' dtype (parquet dictionary encoding). We KEEP them
    # categorical on purpose: casting 15M rows to plain strings materializes millions of Python
    # objects and blows past 16 GB (the Stage 4 lesson). Categorical .isin / groupby / value_counts
    # all work fine, and the whole frame stays a few hundred MB.
    df = pd.read_parquet(STRUCTURED_FILE)
    df = df.drop(columns=[c for c in ["Complaint ID"] if c in df.columns])  # unused in the app

    # Build the 3-level relief grouping WITHOUT materializing strings: remap on the category
    # codes only (4 output buckets), so this stays integer-array work.
    resp = df["Company response to consumer"]
    if not isinstance(resp.dtype, pd.CategoricalDtype):
        resp = resp.astype("category")
    relief_for_cat = np.array(
        [RELIEF_MAP.get(c, "Other / in progress") for c in resp.cat.categories], dtype=object
    )
    group_labels, group_idx = np.unique(relief_for_cat, return_inverse=True)
    codes = resp.cat.codes.to_numpy()
    new_codes = np.where(codes >= 0, group_idx[codes.clip(min=0)], -1).astype("int64")
    df["relief3"] = pd.Categorical.from_codes(new_codes, categories=list(group_labels))
    return df


@st.cache_data(show_spinner="Loading templated cube (once) ...")
def load_cube() -> pd.DataFrame:
    return pd.read_parquet(CUBE_FILE)


@st.cache_data(show_spinner="Indexing companies (once) ...")
def company_list(_df: pd.DataFrame) -> list[str]:
    vc = _df["Company"].value_counts()
    return vc.index.dropna().tolist()  # sorted by volume, most complaints first


# ----------------------------------------------------------------------------- load
if not STRUCTURED_FILE.exists() or not CUBE_FILE.exists():
    st.error(
        "Aggregate files missing. Run first:\n\n"
        "    python scripts/build_dashboard_aggregates.py\n"
    )
    st.stop()

df = load_structured()
cube = load_cube()
companies_by_volume = company_list(df)

# ----------------------------------------------------------------------------- sidebar (filters)
st.sidebar.title("Filters")
st.sidebar.caption("These behave like Power BI slicers — they cross-filter every chart at once.")

sel_cats = st.sidebar.multiselect("Category", CATEGORIES, default=CATEGORIES)
if not sel_cats:
    sel_cats = CATEGORIES  # empty == all

yr_min, yr_max = int(df["year"].min()), int(df["year"].max())
sel_years = st.sidebar.slider("Year range", yr_min, yr_max, (yr_min, yr_max))

st.sidebar.caption(f"{len(companies_by_volume):,} companies — leave empty for ALL. Type to search.")
sel_companies = st.sidebar.multiselect(
    "Company (optional)", companies_by_volume, default=[],
    help="Sorted by complaint volume. Empty = all companies.",
)

# ----------------------------------------------------------------------------- apply filters (LIVE)
mask = (
    df["category"].isin(sel_cats)
    & df["year"].between(sel_years[0], sel_years[1])
)
if sel_companies:
    mask &= df["Company"].isin(sel_companies)
fdf = df[mask]

# same filter applied to the precomputed cube (so the templated view cross-filters identically)
cube_mask = (
    cube["category"].isin(sel_cats)
    & cube["year"].between(sel_years[0], sel_years[1])
)
if sel_companies:
    cube_mask &= cube["company"].isin(sel_companies)
fcube = cube[cube_mask]

# ----------------------------------------------------------------------------- header
st.title("📊 Consumer Complaint Analytics Dashboard")
st.caption(
    "CFPB consumer-complaint dataset — four product categories. "
    "Reference blueprint for the Power BI build. "
    "Structured views are computed live; the templated-dispute view reads a precomputed cube."
)
if fdf.empty:
    st.warning("No rows match the current filters.")
    st.stop()

tab_overview, tab_trends, tab_templated, tab_companies, tab_resolution = st.tabs(
    ["Overview", "Trends over time", "Templated dispute-mill", "Companies / Bureaus", "Resolution outcomes"]
)

# ============================================================================= TAB 1 — OVERVIEW
with tab_overview:
    n = len(fdf)
    narr_pct = fdf["has_narrative"].mean()
    n_companies = fdf["Company"].nunique()
    y0, y1 = int(fdf["year"].min()), int(fdf["year"].max())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total complaints", f"{n:,}")
    c2.metric("With published narrative", f"{narr_pct:.1%}")
    c3.metric("Distinct companies", f"{n_companies:,}")
    c4.metric("Year span", f"{y0}–{y1}")

    st.subheader("Category share")
    cc = fdf["category"].value_counts()
    cat_counts = cc[cc > 0].rename_axis("category").reset_index(name="complaints")
    cat_counts["share"] = cat_counts["complaints"] / cat_counts["complaints"].sum()
    chart = (
        alt.Chart(cat_counts)
        .mark_bar()
        .encode(
            x=alt.X("complaints:Q", title="Complaints"),
            y=alt.Y("category:N", sort="-x", title=None),
            color=alt.Color("category:N", legend=None),
            tooltip=["category", alt.Tooltip("complaints:Q", format=","), alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Credit reporting dominates the four-category subset (~86% with no filters) — the severe "
        "class imbalance handled throughout the project."
    )

# ============================================================================= TAB 2 — TRENDS
with tab_trends:
    st.subheader("Complaint volume over time")
    grain = st.radio("Time grain", ["Yearly", "Monthly"], horizontal=True)

    if grain == "Yearly":
        ts = fdf.groupby(["year", "category"], observed=True).size().reset_index(name="complaints")
        ts = ts.rename(columns={"year": "period"})
        ts["period"] = ts["period"].astype(int)
        x_enc = alt.X("period:O", title="Year")
    else:
        ts = (
            fdf.groupby(["year_month", "category"], observed=True).size().reset_index(name="complaints")
        )
        ts = ts.rename(columns={"year_month": "period"})
        x_enc = alt.X("period:T", title="Month")

    line = (
        alt.Chart(ts)
        .mark_line(point=False)
        .encode(
            x=x_enc,
            y=alt.Y("complaints:Q", title="Complaints"),
            color=alt.Color("category:N", title="Category"),
            tooltip=["category", "period", alt.Tooltip("complaints:Q", format=",")],
        )
        .properties(height=380)
    )
    st.altair_chart(line, use_container_width=True)
    st.caption(
        "The credit-reporting surge (≈604k in 2022 → ≈4.81M in 2025 with no filters) dwarfs the other "
        "three categories. ⚠️ 2026 is a PARTIAL year (data through mid-2026) — the final point is not a real decline."
    )

# ============================================================================= TAB 3 — TEMPLATED
with tab_templated:
    st.subheader("Templated ('dispute-mill') share of narratives over time")
    st.caption(
        "A complaint is **templated** when its cleaned narrative text is an exact duplicate of at least "
        "one other complaint (mass-filed boilerplate); **unique** when its text appears only once. "
        "This view reads the precomputed cube. Note: published narratives only exist from **2015**, and "
        "recent months lag (2026 is incomplete) — so 2026 is excluded by default."
    )
    include_2026 = st.checkbox("Include partial 2026", value=False)

    tc = fcube.copy()
    if not include_2026:
        tc = tc[tc["year"] < 2026]

    if tc.empty:
        st.info("No narrative rows match the current filters in the templated cube.")
    else:
        by = tc.groupby(["year", "category"], observed=True).apply(
            lambda d: pd.Series(
                {"narr": int(d["n"].sum()), "templated": int(d.loc[d["is_templated"], "n"].sum())}
            ),
            include_groups=False,
        ).reset_index()
        by["templated_share"] = by["templated"] / by["narr"]

        share_line = (
            alt.Chart(by)
            .mark_line(point=True)
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("templated_share:Q", title="Templated share", axis=alt.Axis(format="%")),
                color=alt.Color("category:N", title="Category"),
                tooltip=["category", "year", alt.Tooltip("templated_share:Q", format=".1%"),
                         alt.Tooltip("templated:Q", format=","), alt.Tooltip("narr:Q", format=",")],
            )
            .properties(height=320)
        )
        st.altair_chart(share_line, use_container_width=True)

        st.markdown("**Absolute templated narrative volume**")
        vol_bar = (
            alt.Chart(by)
            .mark_bar()
            .encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("templated:Q", title="Templated complaints", stack=True),
                color=alt.Color("category:N", title="Category"),
                tooltip=["category", "year", alt.Tooltip("templated:Q", format=",")],
            )
            .properties(height=260)
        )
        st.altair_chart(vol_bar, use_container_width=True)
        st.caption(
            "With no filters, credit-reporting templated share climbs ~11% (2015) → ~77% (2025): strong "
            "evidence the surge is automated mass-filing rather than distinct consumer harm."
        )

# ============================================================================= TAB 4 — COMPANIES
with tab_companies:
    st.subheader("Top companies by complaint volume")
    top_n = st.slider("How many companies", 5, 40, 15)
    vc = fdf["Company"].value_counts()
    vc = vc[vc > 0].head(top_n)
    comp = vc.rename_axis("Company").reset_index(name="complaints")
    comp["Company"] = comp["Company"].astype("string")  # drop unused categories for clean axis
    comp["is_bureau"] = comp["Company"].isin(BUREAUS)
    bar = (
        alt.Chart(comp)
        .mark_bar()
        .encode(
            x=alt.X("complaints:Q", title="Complaints"),
            y=alt.Y("Company:N", sort="-x", title=None),
            color=alt.Color(
                "is_bureau:N",
                title="Credit bureau",
                scale=alt.Scale(domain=[True, False], range=["#d62728", "#1f77b4"]),
            ),
            tooltip=["Company", alt.Tooltip("complaints:Q", format=",")],
        )
        .properties(height=28 * top_n + 40)
    )
    st.altair_chart(bar, use_container_width=True)
    st.caption(
        "⚠️ This is raw VOLUME, not a complaint RATE — we have no per-company market-share denominator, "
        "so a big bar means many complaints, not necessarily a worse company. The three credit bureaus "
        "(TransUnion, Equifax, Experian) are highlighted in red."
    )

# ============================================================================= TAB 5 — RESOLUTION
with tab_resolution:
    st.subheader("How complaints are resolved")
    view = st.radio(
        "Outcome view", ["3-level relief grouping", "Raw response categories"], horizontal=True
    )
    outcome_col = "relief3" if view.startswith("3-level") else "Company response to consumer"

    res = (
        fdf.groupby(["category", outcome_col], observed=True).size().reset_index(name="complaints")
    )
    totals = res.groupby("category", observed=True)["complaints"].transform("sum")
    res["share"] = res["complaints"] / totals

    sort_arg = RELIEF_ORDER if outcome_col == "relief3" else "ascending"
    stacked = (
        alt.Chart(res)
        .mark_bar()
        .encode(
            x=alt.X("share:Q", title="Share within category", axis=alt.Axis(format="%"), stack="normalize"),
            y=alt.Y("category:N", title=None),
            color=alt.Color(f"{outcome_col}:N", title="Outcome", sort=sort_arg),
            tooltip=["category", outcome_col, alt.Tooltip("complaints:Q", format=","),
                     alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=200)
    )
    st.altair_chart(stacked, use_container_width=True)
    if outcome_col == "relief3":
        st.caption(
            "3-level grouping (Stage 6): 'Closed with explanation' is folded into **No relief**; "
            "'In progress' / missing / 'Untimely' / legacy codes are shown as **Other / in progress** "
            "and excluded from the formal 3-level analysis. Credit card wins monetary relief far more "
            "often (relatively); credit reporting resolves mostly via non-monetary file corrections."
        )

    st.markdown("**Timely response rate by category**")
    timely = (
        fdf.assign(timely=fdf["Timely response?"].eq("Yes"))
        .groupby("category", observed=True)["timely"]
        .mean()
        .reset_index()
    )
    timely_bar = (
        alt.Chart(timely)
        .mark_bar()
        .encode(
            x=alt.X("timely:Q", title="Timely %", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0.9, 1.0])),
            y=alt.Y("category:N", sort="-x", title=None),
            tooltip=["category", alt.Tooltip("timely:Q", format=".2%")],
        )
        .properties(height=180)
    )
    st.altair_chart(timely_bar, use_container_width=True)
    st.caption("Overall ~99.5% timely with no filters; debt collection is the worst (~3.5% late).")

# ----------------------------------------------------------------------------- footer
st.divider()
st.caption(
    f"Showing {len(fdf):,} of {len(df):,} complaints under current filters. "
    "Structured views computed live; templated view from precomputed cube. "
    "Power BI note: cross-filtering here is one-directional (sidebar → charts); "
    "Power BI adds click-to-filter between visuals."
)
