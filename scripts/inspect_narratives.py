"""
Stage 1 follow-up - count narrative-bearing rows in the FOUR-category subset.

Applies the CONFIRMED category mapping:
  - Credit reporting  <- collapse of 3 lineage strings
  - Debt collection   <- 'Debt collection'
  - Mortgage          <- 'Mortgage'
  - Credit card       <- 'Credit card' + 'Credit card or prepaid card',
                         EXCLUDING prepaid-type Sub-products.

MEMORY-SAFE: pandas chunked reading of ONLY 3 columns (Product, Sub-product,
narrative). We never store narrative text - we only test non-null per chunk,
tally, and discard. No full-file load.

Run:  python scripts/inspect_narratives.py
"""

import os
import time
import pandas as pd

BASE = r"D:\Code\projects\Consumer_complaint_analytics"
DATA = os.path.join(BASE, "complaints.csv", "complaints.csv")

PRODUCT_COL = "Product"
SUBPRODUCT_COL = "Sub-product"
NARR_COL = "Consumer complaint narrative"

# --- confirmed mapping -----------------------------------------------------
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


def assign_category(prod: pd.Series, sub: pd.Series) -> pd.Series:
    """Vectorized mapping of (Product, Sub-product) -> one of the 4 categories or NA."""
    cat = pd.Series(pd.NA, index=prod.index, dtype="object")
    cat[prod.isin(CREDIT_REPORTING)] = "Credit reporting"
    cat[prod == "Debt collection"] = "Debt collection"
    cat[prod == "Mortgage"] = "Mortgage"
    # credit card, minus prepaid sub-products
    is_cc = prod.isin(CREDIT_CARD) & ~sub.isin(PREPAID_SUBPRODUCTS)
    cat[is_cc] = "Credit card"
    return cat


CATS = ["Credit reporting", "Debt collection", "Mortgage", "Credit card"]
CHUNK = 250_000

total_rows = pd.Series(0, index=CATS, dtype="int64")
narr_rows = pd.Series(0, index=CATS, dtype="int64")
# also track prepaid rows removed from the credit-card strings
cc_strings_total = 0
cc_after_exclusion = 0

print("Counting narrative-bearing rows in the 4-category subset (pandas chunked)...")
t0 = time.time()
n_chunks = 0
reader = pd.read_csv(
    DATA,
    usecols=[PRODUCT_COL, SUBPRODUCT_COL, NARR_COL],
    dtype={PRODUCT_COL: "string", SUBPRODUCT_COL: "string", NARR_COL: "string"},
    encoding="utf-8",
    chunksize=CHUNK,
)
for chunk in reader:
    n_chunks += 1
    prod = chunk[PRODUCT_COL]
    sub = chunk[SUBPRODUCT_COL]
    narr = chunk[NARR_COL]

    cat = assign_category(prod, sub)
    in_subset = cat.notna()

    # narrative present = not null AND not empty/whitespace
    has_narr = narr.notna() & (narr.str.strip().str.len() > 0)

    sub_cat = cat[in_subset]
    sub_has = has_narr[in_subset]

    total_rows = total_rows.add(sub_cat.value_counts().reindex(CATS, fill_value=0), fill_value=0)
    narr_rows = narr_rows.add(sub_cat[sub_has].value_counts().reindex(CATS, fill_value=0), fill_value=0)

    # credit-card prepaid accounting
    cc_mask = prod.isin(CREDIT_CARD)
    cc_strings_total += int(cc_mask.sum())
    cc_after_exclusion += int((cc_mask & ~sub.isin(PREPAID_SUBPRODUCTS)).sum())

total_rows = total_rows.astype("int64")
narr_rows = narr_rows.astype("int64")

print(f"(streamed {n_chunks} chunks in {time.time()-t0:.1f}s)\n")

grand_total = int(total_rows.sum())
grand_narr = int(narr_rows.sum())

print(f"{'Category':<18} {'rows':>12} {'with narrative':>16} {'narr %':>8}")
print("-" * 58)
for c in CATS:
    t = int(total_rows[c])
    n = int(narr_rows[c])
    pct = (n / t * 100) if t else 0
    print(f"{c:<18} {t:>12,} {n:>16,} {pct:>7.1f}%")
print("-" * 58)
print(f"{'TOTAL':<18} {grand_total:>12,} {grand_narr:>16,} "
      f"{(grand_narr/grand_total*100):>7.1f}%")

print("\nCredit-card prepaid accounting:")
print(f"  rows under the 2 credit-card Product strings : {cc_strings_total:,}")
print(f"  kept after prepaid exclusion                 : {cc_after_exclusion:,}")
print(f"  dropped as prepaid/wallet/gift/etc.          : {cc_strings_total - cc_after_exclusion:,}")
