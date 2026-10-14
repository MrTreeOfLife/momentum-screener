"""
Momentum Swing Screener (US + Europa)
=====================================
Sucht Aktien mit starkem Momentum für Swing Trades (wenige Tage bis ca. 1 Monat).

Bausteine:
  1. Relative Stärke    : Performance über 1, 3 und 6 Monate im Vergleich zum Index (Rang 1 bis 99)
  2. Trend Template     : die 8 Trendkriterien nach Mark Minervini
  3. Hochs + Volumen    : Nähe zum 52-Wochen-Hoch und erhöhtes Handelsvolumen
  4. Enge Konsolidierung: Schwankung und Volumen trocknen aus (VCP-Idee), mit Trade-Plan
  5. Marktampel         : Ist der Gesamtmarkt gerade gut für Momentum-Käufe?
  6. Zahlentermine      : Warnung, wenn eine Firma in den nächsten 14 Tagen Zahlen meldet
  7. Wochenvergleich    : Was ist neu, was ist rausgefallen?
  8. Erfolgskontrolle   : Wie haben sich frühere Kandidaten entwickelt (inkl. Stop-Simulation)?

Ergebnisse (Ordner results/):
  watchlist.md          gut lesbare Watchlist (auch am Handy)
  watchlist.csv         alle bewerteten Aktien
  dashboard.json        Daten für das Dashboard
  market.json           Marktampel je Region und Wechselkurse
  changes.json          Wochenvergleich
  track_record.json     Erfolgskontrolle aller bisherigen Kandidaten
  picks.csv             Archiv aller Watchlist-Kandidaten (wird fortgeschrieben)
  history/DATUM.csv     Archiv jedes Laufs

Hinweis: Kein Anlagerat. Das Tool zeigt nur Kennzahlen aus der Vergangenheit, keine Garantie.
"""

import glob
import io
import os
import re
import sys
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------
# EINSTELLUNGEN (hier kannst du Dinge anpassen)
# --------------------------------------------------------------------------
TOP_US = 12             # Anzahl US-Aktien in der Watchlist
TOP_EU = 8              # Anzahl Europa-Aktien in der Watchlist
MIN_PRICE = 3.0         # Mindestkurs (filtert "Pennystocks")
MIN_TURNOVER_US = 5e6   # Mindest-Tagesumsatz in USD (Liquidität)
MIN_TURNOVER_EU = 1e6   # Mindest-Tagesumsatz in EUR (bzw. GBP/CHF)
MIN_ATR_PCT = 0.008     # Mindest-Schwankung pro Tag. Darunter meist Übernahmekandidaten
MIN_TT_PICK = 7         # Mindestens so viele der 8 Trend-Template-Kriterien für die Watchlist
EARNINGS_WARN_DAYS = 14 # Warnung, wenn Zahlen innerhalb so vieler Tage kommen
EARNINGS_CHECK_TOP = 30 # Für so viele Top-Aktien je Region werden Zahlentermine abgefragt
REWARD_RISK = 2.0       # Kursziel im Trade-Plan = Einstieg + 2 × Risiko
TRACK_DAYS = (5, 10, 20)  # Erfolgskontrolle nach so vielen Handelstagen
WEIGHTS = {"rs": 0.50, "trend": 0.25, "high": 0.25}  # Gewichtung im Gesamtscore

# Aktienlisten (werden jede Woche frisch von Wikipedia geholt)
# (Region, URL, Börsen-Endung falls die Liste keine hat)
SOURCES = [
    ("US", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", ""),
    ("US", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", ""),
    ("US", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", ""),
    ("US", "https://en.wikipedia.org/wiki/Nasdaq-100", ""),
    ("EU", "https://en.wikipedia.org/wiki/DAX", ".DE"),
    ("EU", "https://en.wikipedia.org/wiki/MDAX", ".DE"),
    ("EU", "https://en.wikipedia.org/wiki/EURO_STOXX_50", ".DE"),
    ("EU", "https://en.wikipedia.org/wiki/CAC_40", ".PA"),
    ("EU", "https://en.wikipedia.org/wiki/FTSE_100_Index", ".L"),
    ("EU", "https://en.wikipedia.org/wiki/Swiss_Market_Index", ".SW"),
]

# Vergleichsindex je Region
BENCHMARK = {"US": "SPY", "EU": "EXSA.DE"}  # S&P 500 ETF / STOXX Europe 600 ETF
BENCHMARK_NAME = {"US": "S&P 500 (SPY)", "EU": "STOXX Europe 600 (EXSA)"}
FX = {"USD": "EURUSD=X", "GBP": "EURGBP=X", "CHF": "EURCHF=X"}  # Wechselkurse für den Positionsrechner

HEADERS = {"User-Agent": "Mozilla/5.0 (momentum-screener; educational use)"}
OUT_DIR = "results"


# --------------------------------------------------------------------------
# 1) AKTIENLISTE LADEN
# --------------------------------------------------------------------------
SECTOR_RULES = [  # (Stichworte, einheitlicher Sektor auf Deutsch)
    (("health", "pharma", "biotech", "medic", "hospital"), "Gesundheit"),
    (("information tech", "technology", "software", "semiconductor", "it services", "electronic"), "Technologie"),
    (("energy", "oil", "gas"), "Energie"),
    (("utilit", "electricity"), "Versorger"),
    (("real estate", "reit", "property"), "Immobilien"),
    (("financ", "bank", "insurance", "investment", "life insurance"), "Finanzen"),
    (("communication", "telecom", "media"), "Kommunikation"),
    (("material", "chemi", "metal", "mining", "packaging", "construction material"), "Rohstoffe & Chemie"),
    (("staples", "defensive", "beverage", "food", "tobacco", "household", "personal care", "retail staples"), "Basiskonsum"),
    (("consumer", "retail", "automo", "apparel", "leisure", "travel", "tourism", "e-commerce",
      "sports", "hotel", "airline", "homebuilding"), "Zyklischer Konsum"),
    (("industr", "aerospace", "defen", "engineering", "logistic", "manufactur", "construction",
      "conglomerate", "airport", "waste", "distribution", "handling", "transport", "support services"), "Industrie"),
]


def normalize_sector(raw):
    s = str(raw).lower()
    if not s or s == "nan":
        return "Sonstige"
    for words, label in SECTOR_RULES:
        if any(w in s for w in words):
            return label
    return "Sonstige"


def currency_of(ticker):
    """Handelswährung aus der Börsen-Endung. London notiert in Pence (GBp)."""
    if ticker.endswith(".L"):
        return "GBp"
    if ticker.endswith(".SW"):
        return "CHF"
    if re.search(r"\.[A-Z]{1,2}$", ticker):
        return "EUR"
    return "USD"


def _find_col(columns, candidates):
    for col in columns:
        name = str(col).strip().lower()
        if name in candidates:
            return col
    return None


def load_universe():
    rows = []
    for region, url, suffix in SOURCES:
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
            tables = pd.read_html(io.StringIO(html))
        except Exception as e:  # Quelle nicht erreichbar -> überspringen
            print(f"[WARN] Liste nicht geladen: {url} ({e})")
            continue

        found = False
        for t in tables:
            if isinstance(t.columns, pd.MultiIndex):
                t.columns = [c[-1] for c in t.columns]
            tick_col = _find_col(t.columns, {"symbol", "ticker", "ticker symbol"})
            if tick_col is None or len(t) < 20:
                continue
            name_col = _find_col(t.columns, {"security", "company", "name"})
            sect_col = _find_col(t.columns, {"gics sector", "sector", "prime standard sector", "industry",
                                             "ftse industry classification benchmark sector"})
            for _, r in t.iterrows():
                raw = re.sub(r"\[.*?\]", "", str(r[tick_col])).strip().upper()
                if not raw or raw == "NAN":
                    continue
                if region == "US":
                    ticker = raw.replace(".", "-")          # BRK.B -> BRK-B
                elif suffix == ".L":
                    ticker = raw.replace(".", "-") + ".L"   # London: BT.A -> BT-A.L
                elif re.search(r"\.[A-Z]{1,2}$", raw):
                    ticker = raw                            # hat schon Endung (ADS.DE)
                else:
                    ticker = raw + suffix                   # MDAX: AIXA -> AIXA.DE
                rows.append({
                    "ticker": ticker,
                    "region": region,
                    "name": str(r[name_col]) if name_col is not None else "",
                    "sector": normalize_sector(r[sect_col]) if sect_col is not None else "Sonstige",
                })
            found = True
            break
        print(f"[INFO] {url.split('/')[-1]}: {'ok' if found else 'keine Tabelle gefunden'}")

    uni = pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)
    print(f"[INFO] Universum: {len(uni)} Aktien "
          f"(US {sum(uni.region == 'US')}, EU {sum(uni.region == 'EU')})")
    return uni


# --------------------------------------------------------------------------
# 2) KURSDATEN LADEN
# --------------------------------------------------------------------------
def download_prices(tickers, chunk=150, min_bars=130):
    """Lädt ca. 14 Monate Tageskurse. Gibt dict ticker -> DataFrame zurück."""
    data = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        df = None
        for attempt in range(3):
            try:
                df = yf.download(part, period="14mo", interval="1d", auto_adjust=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"[WARN] Download-Fehler, neuer Versuch ({e})")
                time.sleep(5)
        if df is None:
            continue
        for t in part:
            try:
                sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
                sub = sub[["High", "Low", "Close", "Volume"]].dropna()
                if len(sub) >= min_bars:
                    sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
                    data[t] = sub
            except Exception:
                pass
        print(f"[INFO] Kurse geladen: {min(i + chunk, len(tickers))}/{len(tickers)}")
    return data


# --------------------------------------------------------------------------
# 3) KENNZAHLEN JE AKTIE
# --------------------------------------------------------------------------
def ret(series, days):
    if len(series) <= days:
        return np.nan
    return series.iloc[-1] / series.iloc[-1 - days] - 1


def true_range(df):
    h, l, c = df["High"], df["Low"], df["Close"]
    return pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)


def compute_metrics(df, bench_close):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(c.iloc[-1])

    # Relative Stärke (Rohwert, der Rang 1-99 entsteht später je Region)
    r1, r3, r6 = ret(c, 21), ret(c, 63), ret(c, 126)
    b1, b3, b6 = ret(bench_close, 21), ret(bench_close, 63), ret(bench_close, 126)
    rs_raw = 0.2 * (r1 - b1) + 0.4 * (r3 - b3) + 0.4 * (r6 - b6)

    # Gleitende Durchschnitte
    sma = lambda n: c.rolling(n).mean()
    sma20, sma50, sma150, sma200 = sma(20), sma(50), sma(150), sma(200)
    s20, s50 = sma20.iloc[-1], sma50.iloc[-1]
    s150 = sma150.iloc[-1] if len(c) >= 150 else np.nan
    s200 = sma200.iloc[-1] if len(c) >= 200 else np.nan
    s200_prev = sma200.iloc[-22] if len(c) >= 222 else np.nan

    high52 = float(h.iloc[-252:].max())
    low52 = float(l.iloc[-252:].min())

    # Minervini Trend Template, Kriterien 1-7 (Kriterium 8 = RS-Rang >= 70 kommt später)
    gt = lambda a, b: bool(not (pd.isna(a) or pd.isna(b)) and a > b)
    tt = {
        "tt1_kurs_ueber_150_200": gt(last, s150) and gt(last, s200),
        "tt2_150_ueber_200": gt(s150, s200),
        "tt3_200_steigt": gt(s200, s200_prev),
        "tt4_50_ueber_150_200": gt(s50, s150) and gt(s50, s200),
        "tt5_kurs_ueber_50": gt(last, s50),
        "tt6_30pct_ueber_tief": bool(last >= 1.30 * low52),
        "tt7_max_25pct_unter_hoch": bool(last >= 0.75 * high52),
    }

    # Hochs + Volumen
    dist_high = last / high52 - 1                         # 0 = am Hoch, -0.10 = 10% darunter
    new_high_5d = bool(h.iloc[-5:].max() >= high52 * 0.999)
    vol50 = v.iloc[-55:-5].mean()
    vol_ratio = v.iloc[-5:].mean() / vol50 if vol50 > 0 else np.nan
    proximity = float(np.clip(1 + dist_high * 5, 0, 1))   # 20% unter Hoch -> 0
    vol_part = float(np.clip((vol_ratio if not np.isnan(vol_ratio) else 0) / 2, 0, 1))
    high_score = 70 * proximity + 30 * vol_part

    # Schwankung (ATR)
    tr = true_range(df)
    atr = float(tr.rolling(14).mean().iloc[-1])
    atr10 = float(tr.iloc[-10:].mean())
    atr50 = float(tr.iloc[-60:-10].mean())

    # Enge Konsolidierung (VCP-Idee): die letzten 10 Tage ruhiger und mit weniger Volumen
    hi10, lo10 = float(h.iloc[-10:].max()), float(l.iloc[-10:].min())
    range10 = (hi10 - lo10) / last
    vol10 = v.iloc[-10:].mean()
    vol_prev50 = v.iloc[-60:-10].mean()
    vol_dry = vol10 / vol_prev50 if vol_prev50 > 0 else np.nan
    contraction = atr10 / atr50 if atr50 > 0 else np.nan
    tight = bool(
        contraction < 0.85 and range10 <= max(0.10, 4 * atr / last)
        and dist_high > -0.15 and (not np.isnan(vol_dry)) and vol_dry < 1.0
    )

    # Trade-Plan: Kauf knapp über dem 10-Tage-Hoch, Stop am 10-Tage-Tief
    # (mindestens 1 ATR, höchstens 2 ATR entfernt), Ziel bei 2:1
    entry = max(hi10 * 1.002, last)
    risk = float(np.clip(entry - lo10, 1 * atr, 2 * atr))
    stop_plan = entry - risk
    target_plan = entry + REWARD_RISK * risk

    # Übernahme-Check: großer Sprung, danach klebt der Kurs am Angebotspreis
    range_med5 = ((h.iloc[-5:] - l.iloc[-5:]) / c.iloc[-5:]).median()
    max_jump20 = c.pct_change().iloc[-20:].abs().max()
    takeover = bool(max_jump20 > 0.15 and range_med5 < 0.012)

    return {
        "price": last,
        "ret_1m": r1, "ret_3m": r3, "ret_6m": r6,
        "rs_raw": rs_raw,
        **tt,
        "above_sma50": gt(last, s50),
        "sma50_rising": gt(s50, sma50.iloc[-11]),
        "dist_high": dist_high,
        "dist_low": last / low52 - 1,
        "new_high_5d": new_high_5d,
        "vol_ratio": vol_ratio,
        "high_score": high_score,
        "atr_pct": atr / last,
        "sma20": s20, "sma50": s50,
        "stop_2atr": last - 2 * atr,
        "tight": tight,
        "range10": range10,
        "contraction": contraction,
        "vol_dry": vol_dry,
        "entry_plan": entry,
        "stop_plan": stop_plan,
        "target_plan": target_plan,
        "risk_pct": risk / entry,
        "takeover": takeover,
        "turnover": float((c.iloc[-20:] * v.iloc[-20:]).mean()),
    }


TT_KEYS = ["tt1_kurs_ueber_150_200", "tt2_150_ueber_200", "tt3_200_steigt", "tt4_50_ueber_150_200",
           "tt5_kurs_ueber_50", "tt6_30pct_ueber_tief", "tt7_max_25pct_unter_hoch", "tt8_rs_ueber_70"]


def score(uni, prices):
    out = []
    for region, bench in BENCHMARK.items():
        if bench not in prices:
            print(f"[WARN] Benchmark {bench} fehlt, Region {region} übersprungen")
            continue
        bclose = prices[bench]["Close"]
        min_turn = MIN_TURNOVER_US if region == "US" else MIN_TURNOVER_EU
        for _, r in uni[uni.region == region].iterrows():
            df = prices.get(r.ticker)
            if df is None:
                continue
            try:
                m = compute_metrics(df, bclose)
            except Exception as e:
                print(f"[WARN] {r.ticker}: {e}")
                continue
            turnover = m["turnover"] / 100 if r.ticker.endswith(".L") else m["turnover"]  # London: Pence
            if (m["price"] < MIN_PRICE or turnover < min_turn or np.isnan(m["rs_raw"])
                    or m["atr_pct"] < MIN_ATR_PCT or m["takeover"]):
                continue
            out.append({**r.to_dict(), **m})

    res = pd.DataFrame(out)
    if res.empty:
        return res
    # RS-Rang 1-99 innerhalb der Region (wie IBD Relative Strength Rating)
    res["rs_rank"] = res.groupby("region")["rs_raw"].rank(pct=True) * 98 + 1
    res["tt8_rs_ueber_70"] = res["rs_rank"] >= 70
    res["tt_pts"] = res[TT_KEYS].sum(axis=1).astype(int)
    res["trend_score"] = res["tt_pts"] / 8 * 100
    res["score"] = (WEIGHTS["rs"] * res["rs_rank"]
                    + WEIGHTS["trend"] * res["trend_score"]
                    + WEIGHTS["high"] * res["high_score"])
    # Aufwärtstrend (für Marktbreite): Kurs über SMA50 und SMA50 steigend
    res["qualifies"] = res["above_sma50"] & res["sma50_rising"]
    res["currency"] = res["ticker"].map(currency_of)
    res = res.sort_values("score", ascending=False).reset_index(drop=True)

    # Watchlist: Aufwärtstrend + mindestens MIN_TT_PICK Trend-Template-Kriterien
    res["pick"] = False
    res["pick_rank"] = np.nan
    for region, top in (("US", TOP_US), ("EU", TOP_EU)):
        idx = res[(res.region == region) & res.qualifies & (res.tt_pts >= MIN_TT_PICK)].head(top).index
        res.loc[idx, "pick"] = True
        res.loc[idx, "pick_rank"] = list(range(1, len(idx) + 1))
    return res


# --------------------------------------------------------------------------
# 4) ZAHLENTERMINE
# --------------------------------------------------------------------------
def next_earnings(ticker, today):
    """Nächster Zahlentermin laut Yahoo (oder None). Nie ein Absturz, nur eine Warnung."""
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
        future = sorted(d for d in dates if isinstance(d, date) and d >= today)
        return future[0] if future else None
    except Exception as e:
        print(f"[WARN] Zahlentermin {ticker}: {e}")
        return None


def add_earnings(res, today):
    res["earnings_date"] = None
    res["days_to_earnings"] = np.nan
    cand = res[res.qualifies].groupby("region").head(EARNINGS_CHECK_TOP).index.union(res[res.pick].index)
    for i in cand:
        d = next_earnings(res.at[i, "ticker"], today)
        if d is not None:
            res.at[i, "earnings_date"] = d.isoformat()
            res.at[i, "days_to_earnings"] = (d - today).days
        time.sleep(0.2)
    res["earn_warn"] = res["days_to_earnings"].between(0, EARNINGS_WARN_DAYS)
    print(f"[INFO] Zahlentermine abgefragt: {len(cand)}, davon bekannt: {res['earnings_date'].notna().sum()}")
    return res


# --------------------------------------------------------------------------
# 5) MARKTAMPEL
# --------------------------------------------------------------------------
def market_regime(res, prices, data_date):
    rows = []
    fx = {}
    for cur, sym in FX.items():
        if sym in prices:
            fx[cur] = float(prices[sym]["Close"].iloc[-1])   # 1 EUR = x USD/GBP/CHF
    for region, bench in BENCHMARK.items():
        if bench not in prices:
            continue
        df = prices[bench]
        c, v = df["Close"], df["Volume"]
        s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
        reg = res[res.region == region]
        breadth50 = float(reg["above_sma50"].mean()) if len(reg) else np.nan
        breadth_trend = float(reg["qualifies"].mean()) if len(reg) else np.nan
        chg = c.pct_change()
        dist_days = int(((chg < -0.002) & (v > v.shift(1))).iloc[-25:].sum())
        checks = {
            "index_ueber_sma50": bool(c.iloc[-1] > s50.iloc[-1]),
            "sma50_steigt": bool(s50.iloc[-1] > s50.iloc[-11]),
            "index_ueber_sma200": bool(len(c) >= 200 and c.iloc[-1] > s200.iloc[-1]),
            "breite_ueber_50pct": bool(breadth50 >= 0.5),
        }
        pts = sum(checks.values())
        level = 2 if pts == 4 else (1 if pts >= 2 else 0)
        if region == "US" and dist_days >= 6:   # viele Verkaufstage im S&P 500: eine Stufe vorsichtiger
            level = max(0, level - 1)
        ampel = ["rot", "gelb", "gruen"][level]
        text = {
            "gruen": "Rückenwind. Momentum-Käufe mit normaler Positionsgröße möglich.",
            "gelb": "Gemischt. Nur die besten Setups, halbe Positionsgröße.",
            "rot": "Gegenwind. Keine neuen Käufe, Kapital schützen.",
        }[ampel]
        rows.append({
            "region": region,
            "index_name": BENCHMARK_NAME[region],
            "index_price": round(float(c.iloc[-1]), 2),
            "index_sma50": round(float(s50.iloc[-1]), 2),
            "index_sma200": round(float(s200.iloc[-1]), 2) if len(c) >= 200 else None,
            "index_ret_1m": round(float(ret(c, 21)), 4),
            **checks,
            "breadth50": round(breadth50, 4),
            "breadth_trend": round(breadth_trend, 4),
            "dist_days": dist_days,
            "punkte": int(pts),
            "ampel": ampel,
            "text": text,
            "fx_usd": round(fx["USD"], 4) if "USD" in fx else None,
            "fx_gbp": round(fx["GBP"], 4) if "GBP" in fx else None,
            "fx_chf": round(fx["CHF"], 4) if "CHF" in fx else None,
            "data_date": data_date,
        })
    return rows


# --------------------------------------------------------------------------
# 6) WOCHENVERGLEICH UND ERFOLGSKONTROLLE
# --------------------------------------------------------------------------
PICK_COLS = ["date", "run_date", "ticker", "name", "region", "currency", "pick_rank", "score",
             "price", "stop_2atr", "atr_pct"]


def load_picks(run_date):
    """Bisherige Kandidaten. Beim ersten Mal aus alten history-Dateien rekonstruiert."""
    path = f"{OUT_DIR}/picks.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    rows = []
    for f in sorted(glob.glob(f"{OUT_DIR}/history/*.csv")):
        d = os.path.basename(f)[:-4]
        if d >= run_date:
            continue
        try:
            h = pd.read_csv(f)
        except Exception:
            continue
        if "pick" in h.columns:
            p = h[h["pick"] == True].copy()
        else:  # alte Version ohne Markierung: Top je Region im Aufwärtstrend
            q = h[h["qualifies"] == True].sort_values("score", ascending=False)
            p = pd.concat([q[q.region == "US"].head(TOP_US), q[q.region == "EU"].head(TOP_EU)])
            p["pick_rank"] = p.groupby("region").cumcount() + 1
        p["date"] = p["data_date"] if "data_date" in p.columns else d
        p["run_date"] = d
        if "currency" not in p.columns:
            p["currency"] = p["ticker"].map(currency_of)
        rows.append(p[[c for c in PICK_COLS if c in p.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=PICK_COLS)


def week_changes(res, picks, run_date):
    """Vergleich der aktuellen Watchlist mit der letzten früheren Watchlist."""
    prev_runs = sorted(d for d in picks["run_date"].astype(str).unique() if d < run_date)
    now = res[res.pick].set_index("ticker")
    if not prev_runs:
        return [], None
    prev_run = prev_runs[-1]
    prev = picks[picks["run_date"].astype(str) == prev_run].set_index("ticker")
    all_px = res.set_index("ticker")["price"]
    rows = []
    for t in now.index.union(prev.index):
        in_now, in_prev = t in now.index, t in prev.index
        status = "geblieben" if in_now and in_prev else ("neu" if in_now else "raus")
        src = now.loc[t] if in_now else prev.loc[t]
        price_now = float(all_px[t]) if t in all_px.index else np.nan
        price_prev = float(prev.loc[t, "price"]) if in_prev else np.nan
        rows.append({
            "ticker": t,
            "name": str(src["name"]),
            "region": str(src["region"]),
            "status": status,
            "rank_now": int(now.loc[t, "pick_rank"]) if in_now else None,
            "rank_prev": int(prev.loc[t, "pick_rank"]) if in_prev else None,
            "score_now": round(float(now.loc[t, "score"]), 1) if in_now else None,
            "score_prev": round(float(prev.loc[t, "score"]), 1) if in_prev else None,
            "ret_seit_vorwoche": round(price_now / price_prev - 1, 4) if in_prev and price_now == price_now else None,
            "vorwoche": prev_run,
        })
    order = {"neu": 0, "geblieben": 1, "raus": 2}
    rows.sort(key=lambda r: (order[r["status"]], r["region"], r["rank_now"] or r["rank_prev"] or 99))
    return rows, prev_run


def track_record(picks, prices):
    """Wie haben sich frühere Kandidaten entwickelt?
    Regel der Simulation: Kauf zum Schlusskurs am Tag der Liste, Stop bei Kurs minus 2 ATR,
    Verkauf spätestens nach 20 Handelstagen. Ergebnis in R (1 R = Abstand zum Stop)."""
    out = []
    for _, p in picks.iterrows():
        t, region = p["ticker"], p["region"]
        bench = BENCHMARK.get(region)
        df = prices.get(t)
        entry, stop = float(p["price"]), float(p["stop_2atr"])
        risk = entry - stop
        row = {k: p[k] for k in ["date", "run_date", "ticker", "name", "region", "pick_rank"] if k in p}
        row.update({"entry": round(entry, 2), "stop": round(stop, 2)})
        if df is None or risk <= 0:
            row.update({"status": "keine Daten"})
            out.append(row)
            continue
        after = df[df.index > pd.Timestamp(p["date"])]
        n = len(after)
        for k in TRACK_DAYS:
            row[f"ret_{k}d"] = round(float(after["Close"].iloc[k - 1] / entry - 1), 4) if n >= k else None
            row[f"excess_{k}d"] = None
        if bench in prices:
            b = prices[bench]["Close"]
            b_entry = b[b.index <= pd.Timestamp(p["date"])]
            b_after = b[b.index > pd.Timestamp(p["date"])]
            for k in TRACK_DAYS:
                if len(b_entry) and len(b_after) >= k and row.get(f"ret_{k}d") is not None:
                    row[f"excess_{k}d"] = round(row[f"ret_{k}d"] - float(b_after.iloc[k - 1] / b_entry.iloc[-1] - 1), 4)
        # Stop-Simulation
        window = after.iloc[:TRACK_DAYS[-1]]
        hit = window[window["Low"] <= stop]
        if len(hit):
            row.update({"status": "Stop", "exit_date": hit.index[0].strftime("%Y-%m-%d"), "r_result": -1.0})
        elif n >= TRACK_DAYS[-1]:
            px = float(window["Close"].iloc[-1])
            row.update({"status": "abgeschlossen", "exit_date": window.index[-1].strftime("%Y-%m-%d"),
                        "r_result": round((px - entry) / risk, 2)})
        elif n > 0:
            px = float(after["Close"].iloc[-1])
            row.update({"status": "läuft", "r_result": round((px - entry) / risk, 2)})
        else:
            row.update({"status": "neu", "r_result": None})
        row["tage"] = int(min(n, TRACK_DAYS[-1]))
        out.append(row)
    return out


# --------------------------------------------------------------------------
# 7) AUSGABE
# --------------------------------------------------------------------------
def pct(x):
    return "" if pd.isna(x) else f"{x * 100:+.1f}%"


def md_table(df):
    lines = ["| # | Ticker | Name | Score | RS | TT | 1M | 3M | vom Hoch | Vol | Plan: Kauf / Stop / Ziel | Zahlen |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        flags = (" 🔥" if r.new_high_5d and r.vol_ratio >= 1.5 else "") + (" 🎯" if r.tight else "")
        earn = ""
        if isinstance(r.earnings_date, str):
            earn = ("⚠️ " if r.earn_warn else "") + r.earnings_date
        lines.append(
            f"| {int(r.pick_rank)} | **{r.ticker}**{flags} | {str(r.name)[:26]} | {r.score:.0f} | {r.rs_rank:.0f} | "
            f"{r.tt_pts}/8 | {pct(r.ret_1m)} | {pct(r.ret_3m)} | {pct(r.dist_high)} | {r.vol_ratio:.1f}× | "
            f"{r.entry_plan:.2f} / {r.stop_plan:.2f} / {r.target_plan:.2f} | {earn} |")
    return "\n".join(lines)


def write_outputs(res, market, changes, prev_run, track, run_date, data_date):
    import json
    os.makedirs(f"{OUT_DIR}/history", exist_ok=True)
    res["data_date"] = data_date
    res.to_csv(f"{OUT_DIR}/watchlist.csv", index=False, float_format="%.4f")
    res.to_csv(f"{OUT_DIR}/history/{run_date}.csv", index=False, float_format="%.4f")

    # Archiv der Kandidaten fortschreiben (ein Lauf pro Tag, spätere Läufe ersetzen frühere)
    p = res[res.pick].copy()
    p["date"], p["run_date"] = data_date, run_date
    old = load_picks(run_date)
    old = old[old["run_date"].astype(str) != run_date]
    pd.concat([old, p[PICK_COLS]], ignore_index=True).to_csv(f"{OUT_DIR}/picks.csv", index=False, float_format="%.4f")

    # Markdown-Watchlist
    sym = {"gruen": "🟢", "gelb": "🟡", "rot": "🔴"}
    amp = " · ".join(f"{sym[r['ampel']]} **{'USA' if r['region'] == 'US' else 'Europa'}**: {r['text']}"
                     for r in market)
    us = res[res.pick & (res.region == "US")]
    eu = res[res.pick & (res.region == "EU")]
    neu = [c["ticker"] for c in changes if c["status"] == "neu"]
    raus = [c["ticker"] for c in changes if c["status"] == "raus"]
    wv = (f"**Neu:** {', '.join(neu) or 'keine'} · **Raus:** {', '.join(raus) or 'keine'} (vs. {prev_run})"
          if prev_run else "Erster Lauf, noch kein Vergleich.")
    md = f"""# 📈 Momentum-Watchlist ({run_date})

**Marktampel:** {amp}

**Wochenvergleich:** {wv}

**Bewertete Aktien:** {len(res)} · **Im Aufwärtstrend:** {int(res.qualifies.sum())} · **Kursdaten bis:** {data_date}

## 🇺🇸 USA (Top {TOP_US})
{md_table(us)}

## 🇪🇺 Europa (Top {TOP_EU})
{md_table(eu)}

## Legende
- **Score:** 0 bis 100 (50% Relative Stärke, 25% Trend Template, 25% Hoch + Volumen)
- **RS:** Relative Stärke 1 bis 99 (99 = stärker als 99% der Aktien der Region)
- **TT:** erfüllte Kriterien des Minervini Trend Template (Watchlist ab {MIN_TT_PICK}/8)
- **Plan:** Kauf knapp über dem 10-Tage-Hoch, Stop am 10-Tage-Tief (1 bis 2 ATR), Ziel bei {REWARD_RISK:.0f}:1
- 🔥 neues 52-Wochen-Hoch mit hohem Volumen · 🎯 enge Konsolidierung (VCP-Idee) · ⚠️ Zahlen in den nächsten {EARNINGS_WARN_DAYS} Tagen
- London-Kurse in Pence

> Kein Anlagerat. Kennzahlen aus der Vergangenheit garantieren keine künftige Entwicklung.
"""
    with open(f"{OUT_DIR}/watchlist.md", "w", encoding="utf-8") as f:
        f.write(md)

    # Daten für das Dashboard (kompakt, gerundet)
    dash_cols = ["ticker", "name", "region", "sector", "currency", "price", "score", "rs_rank", "ret_1m",
                 "ret_3m", "ret_6m", "dist_high", "dist_low", "vol_ratio", "tt_pts", *TT_KEYS, "atr_pct",
                 "stop_2atr", "sma20", "sma50", "new_high_5d", "qualifies", "pick", "pick_rank", "tight",
                 "range10", "contraction", "vol_dry", "entry_plan", "stop_plan", "target_plan", "risk_pct",
                 "earnings_date", "days_to_earnings", "earn_warn"]
    dj = res[dash_cols].copy()
    dj["rs_rank"] = dj["rs_rank"].round().astype(int)
    for c in ["price", "stop_2atr", "sma20", "sma50", "vol_ratio", "entry_plan", "stop_plan", "target_plan",
              "contraction", "vol_dry"]:
        dj[c] = dj[c].astype(float).round(2)
    for c in ["ret_1m", "ret_3m", "ret_6m", "dist_high", "dist_low", "atr_pct", "range10", "risk_pct"]:
        dj[c] = dj[c].astype(float).round(4)
    dj["score"] = dj["score"].round(1)
    dj["stand"] = run_date
    dj["data_date"] = data_date
    dj.to_json(f"{OUT_DIR}/dashboard.json", orient="records", force_ascii=False)

    def clean(v):
        """numpy/pandas-Werte in normales JSON übersetzen (NaN -> null)."""
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, (float, np.floating)):
            return None if np.isnan(v) else float(v)
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return None
        return str(v)

    def dump(name, rows):
        with open(f"{OUT_DIR}/{name}", "w", encoding="utf-8") as f:
            json.dump([{k: clean(v) for k, v in r.items()} for r in rows], f, ensure_ascii=False)

    dump("market.json", market)
    dump("changes.json", changes)
    dump("track_record.json", track)
    print(f"[INFO] Watchlist geschrieben: {len(us)} US, {len(eu)} EU · "
          f"Wochenvergleich: {len(changes)} Zeilen · Erfolgskontrolle: {len(track)} Kandidaten")


def main(run_date=None):
    today = date.fromisoformat(run_date) if run_date else datetime.now(timezone.utc).date()
    run_date = today.isoformat()
    uni = load_universe()
    if uni.empty:
        sys.exit("[ERROR] Keine Aktienliste geladen.")

    picks_before = load_picks(run_date)
    extra = [t for t in picks_before["ticker"].unique() if t not in set(uni.ticker)] if len(picks_before) else []
    tickers = list(uni.ticker) + extra + list(BENCHMARK.values())
    prices = download_prices(tickers)
    prices.update(download_prices(list(FX.values()), min_bars=5))

    res = score(uni, prices)
    if res.empty:
        sys.exit("[ERROR] Keine Aktie bewertet (Kursdaten fehlen?).")
    data_date = prices[BENCHMARK["US"]].index[-1].strftime("%Y-%m-%d") if BENCHMARK["US"] in prices else run_date

    res = add_earnings(res, today)
    market = market_regime(res, prices, data_date)
    changes, prev_run = week_changes(res, picks_before, run_date)
    track = track_record(picks_before[picks_before["run_date"].astype(str) < run_date], prices)
    write_outputs(res, market, changes, prev_run, track, run_date, data_date)


if __name__ == "__main__":
    main()
