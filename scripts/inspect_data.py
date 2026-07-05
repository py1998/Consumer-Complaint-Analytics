"""
Stage 1 - Safe inspection of the raw CFPB complaints CSV.


Run:  python scripts/inspect_data.py
"""

import os
import sys
import csv
import time

import pandas as pd
import dask
import dask.dataframe as dd

# ---------------------------------------------------------------------------
# 0. Locate the file (folder named complaints.csv contains a file of same name)
# ---------------------------------------------------------------------------
BASE = r"D:\Code\projects\Consumer_complaint_analytics"
DATA = os.path.join(BASE, "complaints.csv", "complaints.csv")

print("=" * 78)
print("STAGE 1 - SAFE DATA INSPECTION")
print("=" * 78)
print(f"File path : {DATA}")
if not os.path.exists(DATA):
    sys.exit(f"ERROR: file not found at {DATA}")
size_bytes = os.path.getsize(DATA)
print(f"File size : {size_bytes:,} bytes  ({size_bytes/1e9:.3f} GB)")

# ---------------------------------------------------------------------------
# 1. Sniff delimiter + encoding from the first ~64 KB only
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[1] DELIMITER / ENCODING SNIFF (first 64 KB only)")
print("-" * 78)
with open(DATA, "rb") as fh:
    head_bytes = fh.read(64 * 1024)

encoding = None
for enc in ("utf-8", "utf-8-sig", "latin-1"):
    try:
        head_text = head_bytes.decode(enc)
        encoding = enc
        break
    except UnicodeDecodeError:
        continue
print(f"Decoded OK with encoding: {encoding}")

first_line = head_text.splitlines()[0]
try:
    dialect = csv.Sniffer().sniff(head_text[:8192])
    delim = dialect.delimiter
except Exception as e:
    delim = ","
    print(f"(Sniffer fell back to ',' : {e})")
print(f"Detected delimiter      : {delim!r}")
print(f"First line (raw header) :\n{first_line}")

# ---------------------------------------------------------------------------
# 2. Column names (header row only)
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[2] COLUMN NAMES (read header only, nrows=0)")
print("-" * 78)
cols = pd.read_csv(DATA, nrows=0, encoding=encoding, sep=delim).columns.tolist()
print(f"Number of columns: {len(cols)}")
for i, c in enumerate(cols):
    print(f"  [{i:2d}] {c!r}")

# ---------------------------------------------------------------------------
# 3 & 4. Small sample (20 rows) + inferred dtypes
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[3] SAMPLE - first 20 data rows (shown transposed; first 3 rows)")
print("-" * 78)
sample = pd.read_csv(DATA, nrows=20, encoding=encoding, sep=delim)
with pd.option_context("display.max_colwidth", 90, "display.max_columns", None):
    print(sample.head(3).T)

print("\n" + "-" * 78)
print("[4] DTYPES as inferred from the 20-row sample (NOTE: a small sample can")
print("    mis-guess types; treated as indicative only)")
print("-" * 78)
print(sample.dtypes)

# ---------------------------------------------------------------------------
# Identify the Product column and any Sub-product column (case-insensitive)
# ---------------------------------------------------------------------------
def find_col(candidates):
    norm = {c.lower().replace(" ", "").replace("-", "").replace("_", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("-", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None

product_col = find_col(["Product"])
subproduct_col = find_col(["Sub-product", "Subproduct", "Sub product"])
print("\n" + "-" * 78)
print("[detect] Product column    :", repr(product_col))
print("[detect] Sub-product column:", repr(subproduct_col))
print("-" * 78)
if product_col is None:
    sys.exit("ERROR: could not find a 'Product' column - cannot continue.")

# ---------------------------------------------------------------------------
# 5. FULL Product (+ Sub-product) value counts via PANDAS CHUNKED reading.
#
#    Why pandas chunks and not Dask: Dask splits the file at raw byte offsets
#    and parses each block alone, which breaks on the embedded newlines inside
#    quoted complaint narratives ("EOF inside string"). Pandas reads the file
#    as ONE continuous stream, so it respects quotes/embedded newlines. By
#    selecting only these 1-2 columns and processing one chunk at a time, peak
#    memory stays tiny. This single pass also yields the TRUE record count.
#    No pre-filtering, no guessing the mapping - everything is shown.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[5] FULL DISTINCT 'Product' (+ Sub-product) COUNTS - pandas chunked, one pass")
print("-" * 78)

usecols = [product_col] + ([subproduct_col] if subproduct_col else [])
NA_SENTINEL = "<<MISSING>>"          # so NaN aligns across chunks
CHUNK = 1_000_000                     # rows per chunk; only these cols are kept

prod_counts = None
sub_counts = None
true_rows = 0
n_chunks = 0
t0 = time.time()
reader = pd.read_csv(
    DATA,
    usecols=usecols,
    dtype={c: "string" for c in usecols},
    encoding=encoding,
    sep=delim,
    chunksize=CHUNK,
)
for chunk in reader:
    n_chunks += 1
    true_rows += len(chunk)
    pc = chunk[product_col].fillna(NA_SENTINEL).value_counts()
    prod_counts = pc if prod_counts is None else prod_counts.add(pc, fill_value=0)
    if subproduct_col:
        sc = chunk[subproduct_col].fillna(NA_SENTINEL).value_counts()
        sub_counts = sc if sub_counts is None else sub_counts.add(sc, fill_value=0)

prod_counts = prod_counts.astype("int64").sort_values(ascending=False)
print(f"(streamed {n_chunks} chunks in {time.time()-t0:.1f}s)\n")
print(f"TRUE total data rows (sum of chunk lengths): {true_rows:,}")
print(f"Distinct Product values: {prod_counts.shape[0]}\n")
print(f"{'count':>14}   {'share':>7}   Product string")
print(f"{'-'*14}   {'-'*7}   {'-'*40}")
for val, cnt in prod_counts.items():
    share = cnt / true_rows * 100
    print(f"{int(cnt):>14,}   {share:6.2f}%   {val!r}")

if subproduct_col:
    sub_counts = sub_counts.astype("int64").sort_values(ascending=False)
    print("\n" + "-" * 78)
    print(f"[5b] FULL DISTINCT '{subproduct_col}' VALUES + COUNTS")
    print("-" * 78)
    print(f"Distinct {subproduct_col} values: {sub_counts.shape[0]}\n")
    print(f"{'count':>14}   {'share':>7}   {subproduct_col} string")
    print(f"{'-'*14}   {'-'*7}   {'-'*40}")
    for val, cnt in sub_counts.items():
        share = cnt / true_rows * 100
        print(f"{int(cnt):>14,}   {share:6.2f}%   {val!r}")
else:
    print("\n[5b] No Sub-product-style column detected; skipping.")

# ---------------------------------------------------------------------------
# 6. Cross-check: raw newline count (constant memory). Because narratives can
#    contain embedded newlines, this is an UPPER BOUND on physical lines; the
#    pandas record count in [5] is the authoritative number of complaints.
# ---------------------------------------------------------------------------
print("\n" + "-" * 78)
print("[6] CROSS-CHECK: raw byte newline-count (constant memory)")
print("-" * 78)
t0 = time.time()
newlines = 0
last_byte = b""
with open(DATA, "rb") as fh:
    while True:
        block = fh.read(16 * 1024 * 1024)  # 16 MB blocks
        if not block:
            break
        newlines += block.count(b"\n")
        last_byte = block[-1:]
physical_rows = newlines - 1
if last_byte and last_byte != b"\n":
    physical_rows += 1
print(f"(counted in {time.time()-t0:.1f}s)")
print(f"Physical lines (minus header): {physical_rows:,}")
print(f"Authoritative complaint count from [5]: {true_rows:,}")
diff = physical_rows - true_rows
print(f"Difference (embedded newlines inside narratives): {diff:,}")

print("\n" + "=" * 78)
print("INSPECTION COMPLETE - no full-file pandas load was performed.")
print("=" * 78)
