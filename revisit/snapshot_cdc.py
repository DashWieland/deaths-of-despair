"""Snapshot two public-domain CDC datasets into data/cdc/.

Both live on data.cdc.gov and one of them (the provisional counts) is revised
every month, so a checksum cannot pin the live endpoint. Instead the pipeline
reads a dated snapshot committed to the repo; re-running this script makes a
new snapshot and records its row count and SHA-256 in data/cdc/SNAPSHOTS.md.

* rpvx-m2md  NCHS model-based drug poisoning death rates by county, 2003-2021
* xkb8-kh2a  VSRR provisional drug overdose death counts by state and drug
"""
import datetime as dt
import hashlib
import io
import urllib.request

import pandas as pd

from common import DATA

DATASETS = {
    "nchs_county_drug_poisoning_model": "rpvx-m2md",
    "vsrr_provisional_overdose": "xkb8-kh2a",
}


def main():
    out = DATA / "cdc"
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    lines = []
    for name, ident in DATASETS.items():
        url = f"https://data.cdc.gov/resource/{ident}.csv?$limit=500000"
        raw = urllib.request.urlopen(url, timeout=300).read()
        df = pd.read_csv(io.BytesIO(raw), dtype=str)
        path = out / f"{name}_{today}.csv"
        df.to_csv(path, index=False, lineterminator="\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"| `{path.name}` | data.cdc.gov `{ident}` | {today} | {len(df):,} | `{digest}` |")
        print(f"{path.name}: {len(df):,} rows, {path.stat().st_size:,} bytes, sha256 {digest}")
    manifest = out / "SNAPSHOTS.md"
    header = ("# CDC snapshots\n\nPublic domain (U.S. Government). Made by `revisit/snapshot_cdc.py`.\n\n"
              "| File | Source | Retrieved | Rows | SHA-256 |\n|---|---|---|---|---|\n")
    body = manifest.read_text(encoding="utf-8").split("|---|---|---|---|---|\n", 1)[1] if manifest.exists() else ""
    manifest.write_text(header + body + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
