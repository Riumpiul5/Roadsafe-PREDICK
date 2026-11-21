"""Download STATS19 Road Safety Open Data from UK DfT.

This script downloads the 'last 5 years' collision dataset, validates that
the required schema columns are present, and stores the raw file in dataset/raw/.
It does NOT fabricate data. If the download fails, it exits with a non-zero code.
"""
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
RAW_DIR = Path("dataset/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Official DfT endpoint for the "last 5 years" collision dataset.
# Source: https://www.gov.uk/government/statistical-data-sets/road-safety-open-data
DATA_URL = (
    "https://data.dft.gov.uk/road-accidents-safety-data/"
    "dft-road-casualty-statistics-collision-last-5-years.csv"
)
OUTPUT_CSV = RAW_DIR / "collision_last_5_years.csv"
MANIFEST_PATH = Path("dataset/MANIFEST.md")

# Minimum columns we need for the ML pipeline (Phase 4+).
# Note: DfT renamed 'accident_*' to 'collision_*' from the 2024 release onwards.
REQUIRED_COLUMNS_MODERN = {
    "collision_index",
    "collision_severity",
    "longitude",
    "latitude",
    "speed_limit",
    "road_type",
    "weather_conditions",
    "road_surface_conditions",
    "light_conditions",
    "urban_or_rural_area",
    "junction_detail",
    "date",
    "time",
}
REQUIRED_COLUMNS_LEGACY = {
    "accident_index",
    "accident_severity",
    "longitude",
    "latitude",
    "speed_limit",
    "road_type",
    "weather_conditions",
    "road_surface_conditions",
    "light_conditions",
    "urban_or_rural_area",
    "junction_detail",
    "date",
    "time",
}


def _build_session() -> requests.Session:
    """Session with retries for flaky connections."""
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _download(url: str, dest: Path) -> None:
    """Stream download with progress. Skips if file already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest} already exists ({dest.stat().st_size / 1e6:.1f} MB).")
        return

    print(f"[download] {url}")
    session = _build_session()
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f"\r  {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB ({pct:.0f}%)", end="")
        print()
    print(f"[ok] Saved to {dest}")


def _validate_schema(path: Path) -> str:
    """Read only the header row and check for expected columns. Returns 'modern' or 'legacy'."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    cols = set(header)

    if REQUIRED_COLUMNS_MODERN.issubset(cols):
        print("[schema] Detected MODERN schema (collision_*).")
        return "modern"
    if REQUIRED_COLUMNS_LEGACY.issubset(cols):
        print("[schema] Detected LEGACY schema (accident_*).")
        return "legacy"

    missing_modern = REQUIRED_COLUMNS_MODERN - cols
    missing_legacy = REQUIRED_COLUMNS_LEGACY - cols
    raise ValueError(
        "Schema validation failed. Missing columns:\n"
        f"  (modern) {missing_modern}\n"
        f"  (legacy) {missing_legacy}\n"
        f"  Actual columns: {header}"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_manifest(schema_type: str, checksum: str) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Dataset Manifest

| Field | Value |
|---|---|
| Source | UK Department for Transport — Road Safety Open Data |
| Landing page | https://www.gov.uk/government/statistical-data-sets/road-safety-open-data |
| Direct download | {DATA_URL} |
| Licence | Open Government Licence v3.0 |
| File | `{OUTPUT_CSV.name}` |
| Size | {OUTPUT_CSV.stat().st_size / 1e6:.2f} MB |
| SHA-256 | `{checksum}` |
| Schema detected | `{schema_type}` |
| Downloaded at | {datetime.now(timezone.utc).isoformat()} |
| Downloaded by | `scripts/download_data.py` |

## Notes
- The "last 5 years" file is a rolling compilation; the exact year range
  depends on when DfT last refreshed the endpoint.
- DfT renamed `accident_*` columns to `collision_*` in the 2024 release.
  The ML pipeline (Phase 4) must normalise both into a unified schema.
- No data was fabricated. If this file is present, it was fetched from the
  URL above.
"""
    MANIFEST_PATH.write_text(content, encoding="utf-8")
    print(f"[manifest] Wrote {MANIFEST_PATH}")


def main() -> int:
    try:
        _download(DATA_URL, OUTPUT_CSV)
        schema_type = _validate_schema(OUTPUT_CSV)
        checksum = _sha256(OUTPUT_CSV)
        _write_manifest(schema_type, checksum)
        print("[done] Phase 2 dataset acquisition successful.")
        return 0
    except requests.exceptions.RequestException as exc:
        print(f"[error] Network failure: {exc}", file=sys.stderr)
        print(
            "If the endpoint is blocked from your network, download manually:\n"
            f"  {DATA_URL}\n"
            f"and save it as {OUTPUT_CSV}, then re-run this script.",
            file=sys.stderr,
        )
        return 1
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())