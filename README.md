# Trase Bot Analyzer — MVP

End-to-end Python job to ingest Funnel export CSV, compute universal KPIs + objective-aware KPIs, join media plan targets, detect overspend/underspend & not-spending, generate rule-based insights, post to Funnel via webhook, and (optionally) send a weekly HTML email snapshot.

## Quick Start

1) **Create your Funnel scheduled export** (daily CSV to a stable URL).  
2) Copy `.env.example` to `.env` and fill in your URLs and credentials.  
3) Put/adjust your media plan in `config/media_plan.csv`.  
4) Run:

```bash
pip install -r requirements.txt
python trase_analyzer.py
```

To schedule daily: use cron/GitHub Actions/Cloud Run/Lambda — the script is idempotent.

## Files
- `trase_analyzer.py` — main job
- `config/objective_kpis.json` — objective→KPIs mapping
- `config/media_plan.csv` — sample media plan (replace with your sheet/export)
- `.env.example` — copy to `.env` and configure
- `requirements.txt` — Python deps