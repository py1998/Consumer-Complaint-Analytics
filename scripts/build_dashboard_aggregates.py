"""
Stage 8 — Dashboard aggregate builder (offline, run ONCE before the Streamlit app).

WHAT THIS PRODUCES:

  1. dash_structured.parquet      -- a COLUMN-TRIMMED copy of complaints_clean.parquet.
                                     Every row kept (this is a trim, NOT an aggregate), so the
                                     Streamlit app can filter it LIVE on any column it holds.
                                     ~250 MB; loads once into the app and is cached.

  2. dash_templated_agg.parquet   -- the "templated dispute-mill" CUBE at grain
                                     [year, category, company, is_templated] -> count n.
                                     This is the ONE genuinely expensive thing (exact-duplicate
                                     detection over 3.3M narrative strings); computed here ONCE,
                                     offline, never inside the live app.
                                     Denominator = narrative-bearing rows with NON-EMPTY clean text.

  3. complaint_templated_flag.parquet -- a per-Complaint-ID status column so Power BI can relate
                                     templated/unique to the main fact table 1:1 and slice ANY
                                     visual by it. FOUR-VALUED, deliberately, so "no narrative" is
                                     never conflated with "not templated":
                                        no_narrative          -- complaint has no published narrative
                                        templated             -- narrative present, cleaned text duplicated (>=2x)
                                        unique                -- narrative present, cleaned text appears once
                                        empty_after_cleaning  -- narrative present but cleaned text empty
                                                                 (876 pure-redaction artifacts; see Stage 3)
                                     (A 5th tiny bucket 'narrative_unmatched' appears only if a
                                      narrative row has a missing/non-joinable Complaint ID.)

"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ----- the only columns the live dashboard needs from the structured (no-text) file -----
STRUCTURED_COLS = [
    "Complaint ID",
    "category",
    "year",
    "year_month",
    "Company",
    "Company response to consumer",
    "Timely response?",
    "has_narrative",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_structured(clean_path: Path, out_path: Path) -> int:
    """Trim complaints_clean.parquet to the columns the app needs. Every row kept."""
    log(f"Reading trimmed columns from {clean_path.name} ...")
    table = pq.read_table(clean_path, columns=STRUCTURED_COLS)
    n = table.num_rows
    log(f"  loaded {n:,} rows x {len(STRUCTURED_COLS)} cols")
    pq.write_table(table, out_path, compression="snappy")
    size_mb = out_path.stat().st_size / 1e6
    log(f"  wrote {out_path.name} ({size_mb:.1f} MB)")
    del table
    return n


def compute_templated_status(ml_path: Path):
    """
    Return a DataFrame with one row per narrative complaint:
        ['Complaint ID', 'category', 'year', 'status']
    where status in {templated, unique, empty_after_cleaning}.

    Exact-duplicate detection is done on narrative_clean using factorize+bincount on
    integer codes -- narrative_clean is read as Arrow large_string to avoid a giant
    Python-object array.
    """
    log(f"Reading narrative columns from {ml_path.name} ...")
    table = pq.read_table(
        ml_path,
        columns=["Complaint ID", "category", "Date received", "narrative_clean"],
    )
    n = table.num_rows
    log(f"  loaded {n:,} narrative rows")

    # narrative_clean as an Arrow-backed pandas Series (large_string -> no object array)
    nc = table.column("narrative_clean").cast(pa.large_string())
    nc_series = pd.Series(pd.array(nc, dtype=pd.ArrowDtype(pa.large_string())))

    log("  factorizing cleaned narratives (exact-duplicate groups) ...")
    codes, _uniques = pd.factorize(nc_series, use_na_sentinel=False)
    group_sizes = np.bincount(codes)
    is_dup = group_sizes[codes] > 1  # True if this exact cleaned text appears >= 2x

    # empties (cleaned text == "") are their own documented bucket, NOT templated
    empty_mask = (nc_series == "").to_numpy()

    status = np.where(
        empty_mask,
        "empty_after_cleaning",
        np.where(is_dup, "templated", "unique"),
    )

    cid = table.column("Complaint ID").to_pandas()  # int64 (may contain nulls -> becomes float/NA)
    cat = table.column("category").to_pandas().astype("string")
    year = pd.to_datetime(table.column("Date received").to_pandas()).dt.year.astype("Int64")

    df = pd.DataFrame(
        {
            "Complaint ID": cid,
            "category": cat,
            "year": year,
            "status": pd.Series(status, dtype="string"),
        }
    )

    # quick provenance counts
    vc = df["status"].value_counts()
    log(f"  narrative status counts: {vc.to_dict()}")
    del table, nc, nc_series, codes, group_sizes, is_dup, empty_mask, status
    return df, n


def build_templated_cube(narr_df: pd.DataFrame, clean_path: Path, out_path: Path) -> pd.DataFrame:
    """
    Cube at grain [year, category, company, is_templated] -> n, over narrative rows whose
    status is templated or unique (empties excluded -- they are not part of the templated-share
    denominator, matching Stage 5).
    Company is joined from the structured file on Complaint ID.
    """
    log("Joining Company onto narrative rows for the templated cube ...")
    company_tbl = pq.read_table(clean_path, columns=["Complaint ID", "Company"])
    company_df = company_tbl.to_pandas()
    company_df["Company"] = company_df["Company"].astype("string")
    del company_tbl

    cube_src = narr_df[narr_df["status"].isin(["templated", "unique"])].copy()
    cube_src = cube_src.merge(company_df, on="Complaint ID", how="left")
    del company_df

    cube_src["is_templated"] = cube_src["status"].eq("templated")
    cube = (
        cube_src.groupby(["year", "category", "company" if "company" in cube_src else "Company",
                          "is_templated"], dropna=False, observed=True)
        .size()
        .reset_index(name="n")
    )
    cube = cube.rename(columns={"Company": "company"})
    cube["category"] = cube["category"].astype("string")
    cube["company"] = cube["company"].astype("string")

    cube.to_parquet(out_path, index=False)
    log(f"  wrote {out_path.name} ({len(cube):,} cube rows, {out_path.stat().st_size/1e6:.2f} MB)")
    del cube_src
    return cube


def build_per_id_flag(narr_df: pd.DataFrame, clean_path: Path, out_path: Path) -> pd.Series:
    """
    Per-Complaint-ID 4-valued status covering ALL complaints, for a 1:1 Power BI relationship.
    Built by LEFT-joining the narrative status onto the full fact-table key set.
    """
    log("Building per-Complaint-ID templated flag (covers ALL complaints) ...")
    key_tbl = pq.read_table(clean_path, columns=["Complaint ID", "has_narrative"])
    keys = key_tbl.to_pandas()
    del key_tbl

    # status lookup keyed by Complaint ID (drop narrative rows with a missing/NA id -- can't join)
    narr_valid = narr_df.dropna(subset=["Complaint ID"]).copy()
    narr_valid["Complaint ID"] = narr_valid["Complaint ID"].astype("int64")
    dup_ids = int(narr_valid["Complaint ID"].duplicated().sum())
    if dup_ids:
        log(f"  WARNING: {dup_ids:,} duplicate Complaint IDs among narrative rows -> keeping first")
        narr_valid = narr_valid.drop_duplicates(subset="Complaint ID", keep="first")
    status_by_id = narr_valid.set_index("Complaint ID")["status"]

    mapped = keys["Complaint ID"].map(status_by_id)
    has_narr = keys["has_narrative"].to_numpy()

    # Unmatched rows: if the complaint HAS a narrative but we couldn't map it -> 'narrative_unmatched'
    # (e.g. missing Complaint ID); otherwise it's genuinely 'no_narrative'.
    fallback = np.where(has_narr, "narrative_unmatched", "no_narrative")
    templated_status = mapped.where(mapped.notna(), pd.Series(fallback, index=keys.index)).astype("string")

    out = pd.DataFrame(
        {"Complaint ID": keys["Complaint ID"], "templated_status": templated_status}
    )
    out.to_parquet(out_path, index=False)
    log(f"  wrote {out_path.name} ({len(out):,} rows, {out_path.stat().st_size/1e6:.2f} MB)")

    vc = out["templated_status"].value_counts(dropna=False)
    log(f"  per-ID status counts: {vc.to_dict()}")

    # diagnostics for the 1:1 PBI relationship
    n_null_id = int(keys["Complaint ID"].isna().sum())
    n_dup_id = int(keys["Complaint ID"].dropna().duplicated().sum())
    log(f"  fact-table key health: {n_null_id:,} null Complaint IDs, {n_dup_id:,} duplicate IDs")
    return vc


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Stage 8 dashboard aggregates.")
    ap.add_argument("--input-dir", default="data/processed")
    ap.add_argument("--output-dir", default="data/processed")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_path = in_dir / "complaints_clean.parquet"
    ml_path = in_dir / "complaints_ml.parquet"
    for p in (clean_path, ml_path):
        if not p.exists():
            raise SystemExit(f"ERROR: required input not found: {p}")

    t0 = time.time()
    log("=== Stage 8 aggregate build START ===")

    # 1. trimmed structured file
    n_structured = build_structured(clean_path, out_dir / "dash_structured.parquet")

    # 2. narrative status (the expensive dedup), then the cube
    narr_df, n_narr = compute_templated_status(ml_path)
    cube = build_templated_cube(narr_df, clean_path, out_dir / "dash_templated_agg.parquet")

    # 3. per-Complaint-ID flag (covers all rows)
    per_id_vc = build_per_id_flag(narr_df, clean_path, out_dir / "complaint_templated_flag.parquet")

    # ---- reconciliation summary ----
    log("=== RECONCILIATION ===")
    log(f"structured rows                : {n_structured:,}  (expect 15,260,793)")
    log(f"narrative rows (ml)            : {n_narr:,}  (expect 3,307,528)")
    templated_total = int(cube.loc[cube['is_templated'], 'n'].sum())
    unique_total = int(cube.loc[~cube['is_templated'], 'n'].sum())
    log(f"cube templated total          : {templated_total:,}")
    log(f"cube unique total             : {unique_total:,}")
    log(f"cube templated+unique         : {templated_total + unique_total:,}  (expect 3,306,652 non-empty)")
    log(f"per-ID flag total             : {int(per_id_vc.sum()):,}  (expect 15,260,793)")

    log(f"=== DONE in {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
