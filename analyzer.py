import os, io, time, json
from datetime import datetime
import pandas as pd
import requests

# Optional (only if you want to auto-upload to S3 and ping Funnel)
USE_S3 = os.getenv("MODE", "sheet_only") == "funnel_webhook"
if USE_S3:
    import boto3
    from botocore.config import Config

# ---------- Helpers ----------
def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}

    if "date" not in cols:
        raise ValueError("CSV must contain a 'date' column.")

    df["date"] = pd.to_datetime(df[cols["date"]], errors="coerce").dt.date

    def num(col, default=0.0):
        if col in cols:
            return pd.to_numeric(df[cols[col]], errors="coerce").fillna(0.0)
        return pd.Series([default] * len(df))

    df["spend"] = num("spend", 0.0)
    df["conversions"] = num("conversions", 0.0)
    df["revenue"] = num("revenue", 0.0)

    # entity (campaign/adset/etc.)
    entity = None
    for k in ["campaign", "campaign name", "campaign_name", "campaign id", "campaign_id", "account", "source"]:
        if k in cols:
            entity = cols[k]
            break
    if entity is None:
        df["entity"] = "ALL"
    else:
        df = df.rename(columns={entity: "entity"})

    df["cpa"] = df.apply(lambda r: (r["spend"] / r["conversions"]) if r["conversions"] else None, axis=1)
    df["roas"] = df.apply(lambda r: (r["revenue"] / r["spend"]) if r["spend"] else None, axis=1)
    return df

def wow_delta(curr, prev):
    if prev in (None, 0) or pd.isna(prev):
        return None
    return (curr - prev) / prev * 100.0

def build_insights(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    g = df.groupby(["date", "entity"], dropna=False).agg(
        spend=("spend", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    g["cpa"] = g.apply(lambda r: (r["spend"]/r["conversions"]) if r["conversions"] else None, axis=1)
    g["roas"] = g.apply(lambda r: (r["revenue"]/r["spend"]) if r["spend"] else None, axis=1)
    g["week"] = pd.to_datetime(g["date"]).dt.to_period("W").apply(lambda r: r.start_time.date())

    w = g.groupby(["week", "entity"], dropna=False).agg(
        spend=("spend", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
    ).reset_index()
    w["cpa"] = w.apply(lambda r: (r["spend"]/r["conversions"]) if r["conversions"] else None, axis=1)
    w["roas"] = w.apply(lambda r: (r["revenue"]/r["spend"]) if r["spend"] else None, axis=1)

    today = datetime.utcnow().date().isoformat()
    weeks = sorted(w["week"].unique())
    if len(weeks) < 2:
        return pd.DataFrame([{
            "date": today, "source": source_name, "scope": "Account", "entity": "ALL",
            "kpi": "INFO", "value": "", "delta_wow": "",
            "insight": "Not enough history for WoW comparison.",
            "recommendation": "Let export run for another week.",
            "priority": "Low", "confidence": 0.9
        }])

    latest, prev = weeks[-1], weeks[-2]
    curr_df = w[w["week"] == latest].set_index("entity")
    prev_df = w[w["week"] == prev].set_index("entity")

    rows = []
    for ent in curr_df.index.unique():
        if ent not in prev_df.index: 
            continue
        for kpi in ["spend", "conversions", "revenue", "cpa", "roas"]:
            cv, pv = curr_df.loc[ent, kpi], prev_df.loc[ent, kpi]
            if pd.isna(cv) or pd.isna(pv):
                continue
            d = wow_delta(float(cv), float(pv))
            if d is None or abs(d) < 15:  # only meaningful moves
                continue
            prio = "High" if abs(d) >= 30 else "Medium"
            rec = ("Scale +20% on winners; cap frequency."
                   if (kpi == "roas" and d > 0) or (kpi == "cpa" and d < 0)
                   else "Audit creatives/audiences/bids; trim budget on laggards.")
            rows.append({
                "date": today, "source": source_name, "scope": "Campaign", "entity": ent,
                "kpi": kpi.upper(), "value": f"{float(cv):.4g}", "delta_wow": round(d, 1),
                "insight": f"{kpi.upper()} changed {d:.1f}% WoW for '{ent}'.",
                "recommendation": rec, "priority": prio, "confidence": 0.8
            })

    if not rows:
        rows.append({
            "date": today, "source": source_name, "scope": "Account", "entity": "ALL",
            "kpi": "INFO", "value": "", "delta_wow": "",
            "insight": "No notable WoW changes detected.",
            "recommendation": "Maintain budgets; keep monitoring daily.",
            "priority": "Low", "confidence": 0.9
        })
    return pd.DataFrame(rows)

def upload_to_s3_and_get_url(local_path: str) -> str:
    """Uploads file and returns a presigned HTTPS URL (valid ~3 days)."""
    bucket = os.environ["S3_BUCKET"]
    prefix = os.getenv("S3_PREFIX", "trase-insights/")
    key = f"{prefix}{os.path.basename(local_path)}"
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"),
                      config=Config(signature_version="s3v4"))
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": "text/csv"})
    url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=60 * 60 * 24 * 3  # 3 days
    )
    return url

def ping_funnel_webhook(file_url: str):
    endpoint = os.environ["FUNNEL_WEBHOOK_ENDPOINT"]
    token = os.environ["FUNNEL_WEBHOOK_TOKEN"]
    headers = {"Content-Type": "application/json", "x-funnel-fileimport-token": token}
    payload = {"files": [{"url": file_url}]}
    r = requests.post(endpoint, headers=headers, json=payload, timeout=60)
    print("Webhook:", r.status_code, r.text)
    r.raise_for_status()

# ---------- Main ----------
def main():
    # SHEETS_CSV_URLS can be JSON (preferred) or single URL string
    raw = os.getenv("SHEETS_CSV_URLS", "[]").strip()
    sources = []
    if raw.startswith("["):
        sources = json.loads(raw)  # e.g. [{"name":"Funnel data","url":"https://..."}]
    elif raw:
        sources = [{"name": "Funnel data", "url": raw}]
    else:
        raise RuntimeError("Set SHEETS_CSV_URLS env (JSON array or single CSV URL).")

    all_frames = []
    for s in sources:
        name, url = s["name"], s["url"]
        print(f"Fetching: {name} -> {url}")
        try:
            df = fetch_csv(url)
            df = ensure_columns(df)
            ins = build_insights(df, name)
            all_frames.append(ins)
        except Exception as e:
            print(f"[WARN] {name}: {e}")

    if not all_frames:
        print("No insights generated."); return

    out = pd.concat(all_frames, ignore_index=True)
    fname = f"trase_ai_insights_{int(time.time())}.csv"
    out.to_csv(fname, index=False)
    print(f"Saved {fname}\nTop rows:\n", out.head(10))

    if USE_S3:
        url = upload_to_s3_and_get_url(fname)
        print("Uploaded to S3:", url)
        ping_funnel_webhook(url)
    else:
        print("MODE=sheet_only (not posting to Funnel).")

if __name__ == "__main__":
    main()
