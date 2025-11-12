"""
src/data_fetchers/corporate_events.py

Scrape corporate announcements for Deepak Nitrite from BSE (and NSE fallback).
Produce CSV of events with date, title, description, category, sentiment and keyword flags.
"""
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import time

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

analyzer = SentimentIntensityAnalyzer()

# Keywords to flag as binary features
KEYWORDS = {
    "shutdown": ["shutdown", "shut down", "plant shut", "plant shutdown"],
    "fire": ["fire", "blast", "accident", "injury"],
    "order": ["order", "purchase order", "purchase", "order received"],
    "merger": ["acquisition", "merger", "amalgamation", "takeover"],
    "dividend": ["dividend"],
    "results": ["quarter", "q1", "q2", "q3", "q4", "results", "financial results", "earnings"],
    "expansion": ["expansion", "capacity", "capex", "commissioning"],
    "recall": ["recall"],
}

def _flag_keywords(text):
    text_l = text.lower()
    flags = {}
    for key, toks in KEYWORDS.items():
        flags[f"event_{key}"] = int(any(tok in text_l for tok in toks))
    return flags

def fetch_bse_announcements(bse_url="https://www.bseindia.com/stock-share-price/deepak-nitrite-ltd/deepakntr/506401/corp-announcements/"):
    """
    Scrape the BSE announcements page for Deepak Nitrite.
    This function parses announcement blocks present in the HTML.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; bot/0.1)"}
    try:
        resp = requests.get(bse_url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        print("BSE fetch failed:", e)
        return pd.DataFrame()

    soup = BeautifulSoup(resp.text, "lxml")
    # BSE page organizes announcements in table rows or divs. We'll attempt common selectors.
    # Look for <table> with announcements or lists
    rows = []
    # try common pattern: table rows in #ContentPlaceHolder1_grvAnn tbody tr
    table = soup.find("table", {"id": re.compile("grvAnn|tblAnn")})
    if table:
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) >= 2:
                date_text = tds[0].get_text(strip=True)
                title = tds[1].get_text(" ", strip=True)
                # sometimes there is a link to details
                link = tds[1].find("a")
                desc = ""
                if link and link.get("href"):
                    detail_url = link.get("href")
                    if detail_url.startswith("/"):
                        detail_url = "https://www.bseindia.com" + detail_url
                    try:
                        dresp = requests.get(detail_url, headers=headers, timeout=10)
                        dsoup = BeautifulSoup(dresp.text, "lxml")
                        desc = dsoup.get_text(" ", strip=True)[:5000]
                        time.sleep(0.2)
                    except Exception:
                        desc = ""
                rows.append({"date_text": date_text, "title": title, "desc": desc})
    else:
        # fallback: search for announcement list items
        ann_sections = soup.find_all(["div", "li"], string=True)
        for s in ann_sections:
            txt = s.get_text(" ", strip=True)
            # very naive approach: pick lines with a date pattern like dd/mm/yyyy or yyyy-mm-dd
            if re.search(r"\d{1,2}/\d{1,2}/\d{4}", txt) or re.search(r"\d{4}-\d{2}-\d{2}", txt):
                # attempt to split into date + title
                rows.append({"date_text": None, "title": txt, "desc": ""})

    # normalize into DataFrame
    events = []
    for r in rows:
        dt = None
        if r.get("date_text"):
            t = r["date_text"].strip()
            # try common date formats
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
                try:
                    dt = datetime.strptime(t, fmt)
                    break
                except Exception:
                    continue
        # fallback: today
        dt = dt or datetime.utcnow()
        title = r.get("title", "").strip()
        desc = r.get("desc", "").strip()
        sent = analyzer.polarity_scores(title + " " + desc)
        flags = _flag_keywords(title + " " + desc)
        rec = {
            "datetime_utc": dt.isoformat(),
            "title": title,
            "description": desc,
            "compound": sent.get("compound", 0.0),
            "neg": sent.get("neg", 0.0),
            "neu": sent.get("neu", 0.0),
            "pos": sent.get("pos", 0.0),
        }
        rec.update(flags)
        events.append(rec)

    df = pd.DataFrame(events)
    if not df.empty:
        out_path = os.path.join(RAW_DIR, "corporate_events_bse.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved BSE corporate events to {out_path} rows={len(df)}")
    else:
        print("No events parsed from BSE page.")
    return df

def fetch_nse_announcements(symbol="DEEPAKNTR"):
    """
    Attempt to fetch NSE corporate announcements via a lightweight endpoint.
    If NSE public endpoints are rate-limited, this will gracefully return empty DF.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; bot/0.1)"}
    # NSE does not offer a public simple RSS for corporate announcements; we attempt the equity announcements page
    # NOTE: NSE often blocks automated scripts; use with care and respect robots.txt
    url = f"https://www.nseindia.com/api/corporate-announcements?symbol={symbol}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        rows = []
        for item in data.get("data", []):
            dt = item.get("date") or item.get("ANN_DATE")
            try:
                dt_parsed = pd.to_datetime(dt)
            except Exception:
                dt_parsed = pd.Timestamp.utcnow()
            title = item.get("title") or item.get("subject") or item.get("description", "")[:200]
            desc = item.get("description", "")
            sent = analyzer.polarity_scores(title + " " + desc)
            flags = _flag_keywords(title + " " + desc)
            rec = {
                "datetime_utc": dt_parsed.isoformat(),
                "title": title,
                "description": desc,
                "compound": sent.get("compound", 0.0),
                "neg": sent.get("neg", 0.0),
                "neu": sent.get("neu", 0.0),
                "pos": sent.get("pos", 0.0),
            }
            rec.update(flags)
            rows.append(rec)
        df = pd.DataFrame(rows)
        if not df.empty:
            out_path = os.path.join(RAW_DIR, "corporate_events_nse.json")
            df.to_csv(out_path, index=False)
            print(f"Saved NSE corporate events to {out_path} rows={len(df)}")
        return df
    except Exception as e:
        print("NSE fetch failed or blocked:", e)
        return pd.DataFrame()

def aggregate_daily_events(save_path=None):
    """
    Combine BSE + NSE parsed CSVs and aggregate to daily features:
     - mean compound sentiment
     - counts per event keyword
    """
    parts = []
    bse_p = os.path.join(RAW_DIR, "corporate_events_bse.csv")
    nse_p = os.path.join(RAW_DIR, "corporate_events_nse.json")
    for p in [bse_p, nse_p]:
        if os.path.exists(p):
            try:
                parts.append(pd.read_csv(p, parse_dates=["datetime_utc"]))
            except Exception:
                continue
    if not parts:
        print("No event files to aggregate.")
        return None
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["datetime_utc"]).dt.date
    agg = df.groupby("date").agg({
        "compound": "mean",
        "title": "count",
        **{k: "sum" for k in df.columns if k.startswith("event_")}
    }).rename(columns={"title": "event_count"})
    agg.index = pd.to_datetime(agg.index)
    agg_path = save_path or os.path.join(RAW_DIR, "corporate_events_daily.csv")
    agg.to_csv(agg_path)
    print(f"Saved aggregated daily events to {agg_path} rows={len(agg)}")
    return agg

if __name__ == "__main__":
    # fetch both sources and aggregate
    bse = fetch_bse_announcements()
    nse = fetch_nse_announcements()
    agg = aggregate_daily_events()
