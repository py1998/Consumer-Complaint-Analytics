"""
Stage 2 - Build the clean FOUR-category subset from the raw 8.2 GB CFPB CSV.

WHAT THIS SCRIPT DOES:
  - The raw file is ~8.2 GB. Loading it whole into pandas would need ~24 GB RAM. So we STREAM it with pandas in chunks and write the filtered
    result straight to disk - never holding more than one chunk in memory.
  - We deliberately do NOT use Dask's CSV reader here. Dask splits a CSV at raw byte
    offsets and parses each block alone; the complaint narratives contain embedded
    newlines inside quoted fields, so Dask fails with "EOF inside string". pandas reads
    the file as ONE continuous stream and parses quoted narratives correctly.
  - We write the output as Parquet (compressed, columnar). Parquet has no line-splitting
    ambiguity, so Dask handles it cleanly. AFTER the subset exists, we DO use Dask - on
    the Parquet - for a genuine aggregation (complaints per category per year). That is
    Dask used correctly, on a format it handles well, not decoratively.

OUTPUTS:
  - data/interim/complaints_4cat.parquet   (all 16 original cols as text + derived `category`)
  - reports/stage2_summary.txt             (counts, file size, runtime, decision rationale)

Run:  python scripts/build_subset.py
"""

import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = r"D:\Code\projects\Consumer_complaint_analytics"
DATA = os.path.join(BASE, "complaints.csv", "complaints.csv")
OUT_PARQUET = os.path.join(BASE, "data", "interim", "complaints_4cat.parquet")
SUMMARY_TXT = os.path.join(BASE, "reports", "stage2_summary.txt")

PRODUCT_COL = "Product"
SUBPRODUCT_COL = "Sub-product"
NARR_COL = "Consumer complaint narrative"
DATE_COL = "Date received"

# ---------------------------------------------------------------------------
# CONFIRMED four-category mapping (user-approved 2026-06-29) - see progress.md
# ---------------------------------------------------------------------------
CREDIT_REPORTING = {
    "Credit reporting or other personal consumer reports",
    "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit reporting",
}
CREDIT_CARD = {"Credit card", "Credit card or prepaid card"}
PREPAID_SUBPRODUCTS = {
    "General-purpose prepaid card", "Government benefit card",
    "Government benefit payment card", "Gift card", "Gift or merchant card",
    "Payroll card", "ID prepaid card", "Student prepaid card", "Transit card",
    "Electronic Benefit Transfer / EBT card", "General purpose card",
    "Other special purpose card", "Mobile or digital wallet", "Mobile wallet",
}
CATS = ["Credit reporting", "Debt collection", "Mortgage", "Credit card"]

# Expected totals from Stage 1 (used for the built-in self-check)
EXPECTED_TOTAL = {
    "Credit reporting": 13_151_288,
    "Debt collection": 1_144_634,
    "Mortgage": 453_550,
    "Credit card": 511_321,
}
EXPECTED_NARR_TOTAL = 3_307_528


def assign_category(prod: pd.Series, sub: pd.Series) -> pd.Series:
    """Vectorized (Product, Sub-product) -> one of the 4 categories, else NA."""
    cat = pd.Series(pd.NA, index=prod.index, dtype="object")
    cat[prod.isin(CREDIT_REPORTING)] = "Credit reporting"
    cat[prod == "Debt collection"] = "Debt collection"
    cat[prod == "Mortgage"] = "Mortgage"
    cat[prod.isin(CREDIT_CARD) & ~sub.isin(PREPAID_SUBPRODUCTS)] = "Credit card"
    return cat


def main():
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_TXT), exist_ok=True)
    if os.path.exists(OUT_PARQUET):
        os.remove(OUT_PARQUET)  # fresh write so re-runs are clean

    # Exact column names/order from the header (read header only)
    cols = pd.read_csv(DATA, nrows=0, encoding="utf-8").columns.tolist()
    out_cols = cols + ["category"]
    # Fixed Parquet schema: every original column as string + category as string.
    # (Type-casting is deferred to Stage 3 cleaning; Stage 2 preserves raw text.)
    schema = pa.schema([(c, pa.string()) for c in out_cols])

    print("=" * 78)
    print("STAGE 2 - BUILD FOUR-CATEGORY SUBSET (pandas chunked -> Parquet)")
    print("=" * 78)
    print(f"Input : {DATA}")
    print(f"Output: {OUT_PARQUET}")
    print(f"Columns kept: {len(out_cols)} (16 original + category)\n")

    total_in = 0
    kept = pd.Series(0, index=CATS, dtype="int64")
    narr = pd.Series(0, index=CATS, dtype="int64")
    n_chunks = 0
    CHUNK = 200_000

    t0 = time.time()
    writer = pq.ParquetWriter(OUT_PARQUET, schema, compression="snappy")
    try:
        reader = pd.read_csv(
            DATA,
            dtype="string",          # every column as text -> no dtype-inference surprises
            encoding="utf-8",
            chunksize=CHUNK,
        )
        for chunk in reader:
            n_chunks += 1
            total_in += len(chunk)

            cat = assign_category(chunk[PRODUCT_COL], chunk[SUBPRODUCT_COL])
            keep = cat.notna()
            if keep.any():
                sub = chunk.loc[keep].copy()
                sub["category"] = cat[keep].astype("string")

                # tally
                vc = sub["category"].value_counts()
                kept = kept.add(vc.reindex(CATS, fill_value=0), fill_value=0)
                has_narr = (
                    sub[NARR_COL].notna() & (sub[NARR_COL].str.strip().str.len() > 0)
                )
                nvc = sub.loc[has_narr, "category"].value_counts()
                narr = narr.add(nvc.reindex(CATS, fill_value=0), fill_value=0)

                # write this filtered chunk as one Parquet row group
                table = pa.Table.from_pandas(
                    sub[out_cols], schema=schema, preserve_index=False
                )
                writer.write_table(table)

            if n_chunks % 10 == 0:
                print(f"  ...{n_chunks} chunks, {total_in:,} rows read, "
                      f"{int(kept.sum()):,} kept  ({time.time()-t0:.0f}s)")
    finally:
        writer.close()

    elapsed = time.time() - t0
    kept = kept.astype("int64")
    narr = narr.astype("int64")
    total_kept = int(kept.sum())
    total_narr = int(narr.sum())
    size_gb = os.path.getsize(OUT_PARQUET) / 1e9

    # ---------------------- summary + self-check --------------------------
    lines = []
    def out(s=""):
        print(s)
        lines.append(s)

    out("\n" + "-" * 78)
    out("RESULT")
    out("-" * 78)
    out(f"Total rows read from CSV : {total_in:,}")
    out(f"Total rows kept (4 cats) : {total_kept:,}")
    out(f"Rows with narrative      : {total_narr:,}")
    out(f"Output Parquet size      : {size_gb:.2f} GB")
    out(f"Elapsed                  : {elapsed:.0f}s over {n_chunks} chunks\n")

    out(f"{'Category':<18}{'kept':>14}{'expected':>14}{'match':>8}"
        f"{'narrative':>14}")
    out("-" * 68)
    all_ok = True
    for c in CATS:
        k = int(kept[c]); e = EXPECTED_TOTAL[c]; ok = (k == e)
        all_ok &= ok
        out(f"{c:<18}{k:>14,}{e:>14,}{('OK' if ok else 'DIFF'):>8}{int(narr[c]):>14,}")
    out("-" * 68)
    tot_e = sum(EXPECTED_TOTAL.values())
    out(f"{'TOTAL':<18}{total_kept:>14,}{tot_e:>14,}"
        f"{('OK' if total_kept == tot_e else 'DIFF'):>8}{total_narr:>14,}")
    narr_ok = (total_narr == EXPECTED_NARR_TOTAL)
    out(f"\nNarrative self-check: got {total_narr:,}, expected {EXPECTED_NARR_TOTAL:,} "
        f"-> {'OK' if narr_ok else 'DIFF'}")
    out(f"SELF-CHECK OVERALL: {'PASS - matches Stage 1' if (all_ok and narr_ok) else 'MISMATCH - investigate'}")

    # ---------------------- Dask on the Parquet ---------------------------
    out("\n" + "-" * 78)
    out("DASK AGGREGATION on the Parquet (complaints per category per year)")
    out("Dask used here because Parquet is columnar with no line-splitting ambiguity -")
    out("the format it handles well, unlike the raw CSV.")
    out("-" * 78)
    import dask.dataframe as dd
    ddf = dd.read_parquet(OUT_PARQUET, columns=["category", DATE_COL])
    ddf["year"] = ddf[DATE_COL].str.slice(0, 4)
    by_year = (
        ddf.groupby(["category", "year"]).size()
        .compute()
        .reset_index(name="count")
    )
    pivot = (by_year.pivot(index="year", columns="category", values="count")
             .fillna(0).astype("int64").sort_index())
    out(pivot.to_string())

    # ---------------------- decision note ---------------------------------
    out("\n" + "-" * 78)
    out("ENGINEERING DECISION (recorded):")
    out("  Filtering the raw CSV: pandas chunked reading (NOT Dask CSV) - Dask's blockwise")
    out("  CSV reader splits at byte offsets and breaks on embedded newlines inside quoted")
    out("  narratives ('EOF inside string'). Aggregating the result: Dask on Parquet, the")
    out("  format it handles cleanly. Right tool for each job.")
    out("=" * 78)

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSummary written to {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
