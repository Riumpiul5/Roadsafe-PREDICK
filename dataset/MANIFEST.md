# Dataset Manifest

| Field | Value |
|---|---|
| Source | UK Department for Transport — Road Safety Open Data |
| Landing page | https://www.gov.uk/government/statistical-data-sets/road-safety-open-data |
| Direct download | https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv |
| Licence | Open Government Licence v3.0 |
| File | `collision_last_5_years.csv` |
| Size | 97.67 MB |
| SHA-256 | `06a03c0e5504175e23ed98e3e0b4767ac61d396c073405a1bddab9339c18f948` |
| Schema detected | `modern` |
| Downloaded at | 2026-11-19T22:50:18.543774+00:00 |
| Downloaded by | `scripts/download_data.py` |

## Notes
- The "last 5 years" file is a rolling compilation; the exact year range
  depends on when DfT last refreshed the endpoint.
- DfT renamed `accident_*` columns to `collision_*` in the 2024 release.
  The ML pipeline (Phase 4) must normalise both into a unified schema.
- No data was fabricated. If this file is present, it was fetched from the
  URL above.
