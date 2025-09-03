
import os
import io
import json
import math
import smtplib
import requests
import numpy as np
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ----------------------
# Config & Helpers
# ----------------------
load_dotenv()

FUNNEL_EXPORT_URL   = os.getenv("FUNNEL_EXPORT_URL", "")
FUNNEL_WEBHOOK_URL  = os.getenv("FUNNEL_WEBHOOK_URL", "")
FUNNEL_WEBHOOK_TOKEN= os.getenv("FUNNEL_WEBHOOK_TOKEN", "")

EMAIL_ENABLED       = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_SMTP_HOST     = os.getenv("EMAIL_SMTP_HOST", "")
EMAIL_SMTP_PORT     = int(os.getenv("EMAIL_SMTP_PORT", "587"))
EMAIL_SMTP_USER     = os.getenv("EMAIL_SMTP_USER", "")
EMAIL_SMTP_PASS     = os.getenv("EMAIL_SMTP_PASS", "")
EMAIL_FROM          = os.getenv("EMAIL_FROM", "Insights Bot <bot@example.com>")
EMAIL_TO            = os.getenv("EMAIL_TO", "")
EMAIL_SUBJECT_TEMPLATE = os.getenv("EMAIL_SUBJECT_TEMPLATE", "[{client}] Weekly Funnel Insights — KPIs & Actions")

WEEK_OVER_WEEK_THRESHOLD_PCT = float(os.getenv("WEEK_OVER_WEEK_THRESHOLD_PCT", "15"))
RATE_VARIANCE_BAND_PCT       = float(os.getenv("RATE_VARIANCE_BAND_PCT", "15"))
NOT_SPENDING_THRESHOLD_PCT   = float(os.getenv("NOT_SPENDING_THRESHOLD_PCT", "20"))

OBJECTIVE_KPIS_PATH = "config/objective_kpis.json"
MEDIA_PLAN_PATH     = "config/media_plan.csv"

RELEVANT_OBJECTIVE_COL = "objective"  # update if your column differs

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _pct_change(new, old):
    if pd.isna(old) or old == 0:
        return np.nan
    return 100.0 * (new - old) / old

def _safe_div(a, b):
    a = _to_float(a)
    b = _to_float(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def _parse_date(s):
    try:
        return pd.to_datetime(s).date()
    except Exception:
        return pd.NaT

# ----------------------
# Ingest
# ----------------------
def fetch_funnel_csv(url: str) -> pd.DataFrame:
    if not url:
        raise ValueError("FUNNEL_EXPORT_URL is empty. Set it in .env")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    return df

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Standard column lowercasing
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure key columns exist (create if missing)
    for col in ["date","client","brand","market","platform","campaign","objective","spend","impressions","clicks","conversions","revenue","reach","video_views"]:
        if col not in df.columns:
            df[col] = np.nan

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    numeric_cols = ["spend","impressions","clicks","conversions","revenue","reach","video_views"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill NaNs for some columns where 0 is sensible
    for col in ["impressions","clicks","conversions","reach","video_views","revenue","spend"]:
        df[col] = df[col].fillna(0.0)

    return df

# ----------------------
# KPIs
# ----------------------
def compute_universal_kpis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CTR"] = df.apply(lambda r: _safe_div(r["clicks"], r["impressions"]), axis=1)
    df["CPC"] = df.apply(lambda r: _safe_div(r["spend"], r["clicks"]), axis=1)
    df["CPA"] = df.apply(lambda r: _safe_div(r["spend"], r["conversions"]), axis=1)
    df["CPM"] = df.apply(lambda r: _safe_div(r["spend"], r["impressions"]) * 1000.0, axis=1)
    df["ROAS"] = df.apply(lambda r: _safe_div(r["revenue"], r["spend"]), axis=1)
    df["CPV"] = df.apply(lambda r: _safe_div(r["spend"], r["video_views"]), axis=1)
    # Optional engagement_rate if your df includes engagements
    if "engagements" in df.columns:
        df["engagement_rate"] = df.apply(lambda r: _safe_div(r["engagements"], r["impressions"]), axis=1)
    else:
        df["engagement_rate"] = np.nan
    return df

def load_objective_map(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def dynamic_kpi_subset(row: pd.Series, obj_map: dict) -> list:
    obj = str(row.get(RELEVANT_OBJECTIVE_COL, "")).strip()
    keys = obj_map.get(obj, obj_map.get("Default", []))
    # return only those present
    present = [k for k in keys if k in row.index]
    return present

# ----------------------
# Weekly Aggregate & WoW
# ----------------------
def weekly_wow(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["week"] = pd.to_datetime(dfx["date"]).dt.to_period("W").apply(lambda p: p.start_time.date())
    group_cols = ["client","brand","market","platform","campaign","objective","week"]
    agg_cols = ["spend","impressions","clicks","conversions","revenue","reach","video_views","CTR","CPC","CPA","CPM","ROAS","CPV","engagement_rate"]
    dfx = dfx.groupby(group_cols, dropna=False)[agg_cols].mean().reset_index()

    # Create lag by campaign
    dfx = dfx.sort_values(group_cols)
    dfx["prev_week"] = dfx.groupby(["client","brand","market","platform","campaign"])["week"].shift(1)
    for col in ["spend","impressions","clicks","conversions","revenue","CTR","CPC","CPA","CPM","ROAS","CPV","engagement_rate"]:
        dfx[f"WoW_{col}"] = dfx.groupby(["client","brand","market","platform","campaign"])[col].pct_change() * 100.0
    return dfx

# ----------------------
# Media Plan Join & Checks
# ----------------------
def load_media_plan(path: str) -> pd.DataFrame:
    mp = pd.read_csv(path)
    mp.columns = [c.strip().lower() for c in mp.columns]
    # types
    for col in ["total_budget","duration_days","daily_cap","target_cpc","target_cpl","target_cpm","target_cpv"]:
        if col in mp.columns:
            mp[col] = pd.to_numeric(mp[col], errors="coerce")
    return mp

def join_media_plan(df_week: pd.DataFrame, mp: pd.DataFrame) -> pd.DataFrame:
    keys = ["client","brand","market","platform","campaign","objective"]
    for k in keys:
        if k not in df_week.columns:
            df_week[k] = np.nan
        if k not in mp.columns:
            mp[k] = np.nan
    merged = pd.merge(df_week, mp, on=keys, how="left", suffixes=("","_plan"))
    return merged

def expected_daily_spend(row):
    if pd.isna(row.get("total_budget")) or pd.isna(row.get("duration_days")) or row["duration_days"] == 0:
        return np.nan
    return row["total_budget"] / row["duration_days"]

def compute_plan_diagnostics(df_week_mp: pd.DataFrame) -> pd.DataFrame:
    df = df_week_mp.copy()
    df["expected_daily"] = df.apply(expected_daily_spend, axis=1)

    # Rate vs target (use only relevant to objective)
    df["rate_vs_target_flag"] = False
    def rate_check(r):
        obj = str(r.get("objective",""))
        band = RATE_VARIANCE_BAND_PCT
        flags = []
        # choose target by objective
        if obj.lower().startswith("awareness"):
            # CPM target
            if not pd.isna(r.get("target_cpm")) and not pd.isna(r.get("CPM")):
                pct = _pct_change(r["CPM"], r["target_cpm"])
                if abs(pct) >= band:
                    flags.append(("CPM", r["CPM"], pct))
        elif obj.lower().startswith("video"):
            if not pd.isna(r.get("target_cpv")) and not pd.isna(r.get("CPV")):
                pct = _pct_change(r["CPV"], r["target_cpv"])
                if abs(pct) >= band:
                    flags.append(("CPV", r["CPV"], pct))
        elif obj.lower().startswith("consid"):
            if not pd.isna(r.get("target_cpc")) and not pd.isna(r.get("CPC")):
                pct = _pct_change(r["CPC"], r["target_cpc"])
                if abs(pct) >= band:
                    flags.append(("CPC", r["CPC"], pct))
        elif obj.lower().startswith("conv"):
            if not pd.isna(r.get("target_cpl")) and not pd.isna(r.get("CPA")):
                pct = _pct_change(r["CPA"], r["target_cpl"])
                if abs(pct) >= band:
                    flags.append(("CPA", r["CPA"], pct))
        return flags

    df["rate_flags"] = df.apply(rate_check, axis=1)

    # Overspend/Underspend vs plan (weekly perspective -> compare spend vs expected_daily * 7 if available)
    def spend_flag(r):
        if pd.isna(r.get("expected_daily")) or pd.isna(r.get("spend")):
            return np.nan
        expected_week = r["expected_daily"] * 7.0
        if expected_week == 0:
            return np.nan
        pct = _pct_change(r["spend"], expected_week)
        return pct
    df["spend_vs_expected_week_pct"] = df.apply(spend_flag, axis=1)

    return df

# ----------------------
# Rules → Insights
# ----------------------
def rules_to_insights(df_week_mp_diag: pd.DataFrame, obj_map: dict) -> pd.DataFrame:
    insights = []
    for _, r in df_week_mp_diag.iterrows():
        # Setup
        scope = "campaign"
        entity = str(r.get("campaign",""))
        date_val = r.get("week", r.get("date", pd.NaT))
        if pd.isna(date_val):
            date_val = pd.Timestamp.today().date()

        # 1) WoW changes on objective-relevant KPIs
        relevant = obj_map.get(str(r.get("objective","")), obj_map.get("Default", []))
        for k in relevant:
            wow_col = f"WoW_{k}" if f"WoW_{k}" in r.index else None
            if wow_col and not pd.isna(r[wow_col]) and abs(r[wow_col]) >= WEEK_OVER_WEEK_THRESHOLD_PCT:
                insights.append({
                    "date": date_val,
                    "scope": scope,
                    "entity_name": entity,
                    "kpi": k,
                    "value": r.get(k, np.nan),
                    "delta_wow": r[wow_col],
                    "insight": f"{k} moved {r[wow_col]:.1f}% WoW.",
                    "recommendation": f"Investigate drivers of {k} change; adjust bids/creatives/audiences accordingly.",
                    "priority": "High" if abs(r[wow_col]) >= (WEEK_OVER_WEEK_THRESHOLD_PCT*1.5) else "Medium",
                    "confidence": 0.7
                })

        # 2) Rate vs target flags
        for kpi, val, pct in r.get("rate_flags", []):
            direction = "above" if pct > 0 else "below"
            rec = {
                "CPM": "Test broader placements and frequency caps; iterate creatives for lower cost reach.",
                "CPV": "Optimize video length/hook; test placements; widen audience if eligible.",
                "CPC": "Improve CTR via creatives and hooks; test landing pages; refine audience.",
                "CPA": "Tighten retargeting; adjust bid strategy; test high-intent audiences/creatives."
            }.get(kpi, "Optimize strategy against target.")
            insights.append({
                "date": date_val,
                "scope": scope,
                "entity_name": entity,
                "kpi": kpi,
                "value": val,
                "delta_wow": np.nan,
                "insight": f"{kpi} is {abs(pct):.1f}% {direction} the target.",
                "recommendation": rec,
                "priority": "High" if abs(pct) >= (RATE_VARIANCE_BAND_PCT*1.5) else "Medium",
                "confidence": 0.75
            })

        # 3) Spend pacing vs expected
        svsp = r.get("spend_vs_expected_week_pct", np.nan)
        if not pd.isna(svsp) and abs(svsp) >= 10:  # 10–20% bands typically
            direction = "over" if svsp > 0 else "under"
            insights.append({
                "date": date_val,
                "scope": scope,
                "entity_name": entity,
                "kpi": "spend_pacing",
                "value": r.get("spend", np.nan),
                "delta_wow": svsp,
                "insight": f"Spend is {abs(svsp):.1f}% {direction} the expected weekly pace.",
                "recommendation": "Reallocate budgets or adjust daily caps to align with plan.",
                "priority": "Medium",
                "confidence": 0.7
            })

    if not insights:
        return pd.DataFrame(columns=["date","scope","entity_name","kpi","value","delta_wow","insight","recommendation","priority","confidence"])
    out = pd.DataFrame(insights)
    # Sort recency first
    out = out.sort_values(["date","priority"], ascending=[False, True])
    return out

# ----------------------
# Webhook Push (Funnel)
# ----------------------
def push_to_funnel_webhook(df: pd.DataFrame) -> dict:
    if not FUNNEL_WEBHOOK_URL:
        return {"status": "skipped", "reason": "FUNNEL_WEBHOOK_URL not set"}
    # Option 1: send JSON rows directly
    payload = {
        "token": FUNNEL_WEBHOOK_TOKEN or None,
        "rows": df.to_dict(orient="records")
    }
    resp = requests.post(FUNNEL_WEBHOOK_URL, json=payload, timeout=60)
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text}
    return {"status_code": resp.status_code, "response": data}

# ----------------------
# Email Snapshot
# ----------------------
def send_email_snapshot(df_insights: pd.DataFrame, client_name: str = "Client") -> None:
    if not EMAIL_ENABLED or not EMAIL_TO:
        return
    top = df_insights.head(10).copy()
    # HTML table
    cols = ["date","entity_name","kpi","value","delta_wow","insight","recommendation","priority"]
    top = top[cols]
    html_table = top.to_html(index=False, justify="left", border=0)
    msg = MIMEMultipart("alternative")
    subject = EMAIL_SUBJECT_TEMPLATE.format(client=client_name)
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    html = f"""
    <html><body>
    <h3>Top Insights</h3>
    {html_table}
    <p style='color:#888'>Generated by Trase Bot Analyzer</p>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
        server.starttls()
        if EMAIL_SMTP_USER:
            server.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASS)
        server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())

# ----------------------
# Two-Client Comparison (mini)
# ----------------------
def build_two_client_comparison(df_week: pd.DataFrame, client_a: str, client_b: str) -> pd.DataFrame:
    subset = df_week[df_week["client"].isin([client_a, client_b])].copy()
    # Choose a few core cols
    keep = ["client","campaign","objective","week","CTR","CPC","CPA","CPM","ROAS","CPV"]
    subset = subset[keep]
    # Pivot by client to compare side-by-side
    piv = subset.pivot_table(index=["campaign","objective","week"],
                             columns="client",
                             values=["CTR","CPC","CPA","CPM","ROAS","CPV"],
                             aggfunc="mean")
    piv = piv.reset_index()
    # Flatten columns
    piv.columns = ["_".join([c for c in col if c]).strip("_") for col in piv.columns.values]
    return piv

# ----------------------
# Main Run
# ----------------------
def main():
    print("→ Fetching Funnel CSV ...")
    raw = fetch_funnel_csv(FUNNEL_EXPORT_URL)
    print(f"Fetched rows: {len(raw)}")

    print("→ Normalizing & computing KPIs ...")
    df = normalize_df(raw)
    df = compute_universal_kpis(df)

    print("→ Weekly aggregate & WoW ...")
    weekly = weekly_wow(df)

    print("→ Load objective map + media plan ...")
    obj_map = load_objective_map(OBJECTIVE_KPIS_PATH)
    mp = load_media_plan(MEDIA_PLAN_PATH)

    print("→ Join media plan & compute diagnostics ...")
    joined = join_media_plan(weekly, mp)
    diag = compute_plan_diagnostics(joined)

    print("→ Rules → insights ...")
    insights = rules_to_insights(diag, obj_map)
    print(f"Insights generated: {len(insights)}")

    # Save outputs locally for inspection
    os.makedirs("out", exist_ok=True)
    df.to_csv("out/daily_with_kpis.csv", index=False)
    weekly.to_csv("out/weekly_agg.csv", index=False)
    diag.to_csv("out/weekly_with_plan_diag.csv", index=False)
    insights.to_csv("out/insights.csv", index=False)

    print("→ Push to Funnel webhook ...")
    wh = push_to_funnel_webhook(insights)
    print("Webhook result:", wh)

    # Optional email
    try:
        send_email_snapshot(insights, client_name="Portfolio")
        print("Email snapshot: done or skipped.")
    except Exception as e:
        print("Email snapshot error:", e)

    print("✓ Done.")

if __name__ == "__main__":
    main()
