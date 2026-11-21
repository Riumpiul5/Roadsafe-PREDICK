# RoadSafe AI — Intelligent Traffic Accident Severity Prediction System

> Predicts accident **severity** (Slight / Serious / Fatal) at the time a
> collision record is created, using STATS19 road-safety features.

## Status

| Phase | Status |
|---|---|
| 1. Project initialisation | ✅ done |
| 2. Dataset acquisition    | ⏳ pending |
| …                          | ⏳ pending |

Full documentation is added incrementally at Phase 15.

## Quick start (Phase 1 only)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
python -c "from app.core.config import get_settings; print(get_settings().app_name)"