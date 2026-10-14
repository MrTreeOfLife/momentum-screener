"""
Momentum Swing Screener (US + Europa)
=====================================
Sucht Aktien mit starkem Momentum für Swing Trades (wenige Tage bis ca. 1 Monat).

Drei Signal-Bausteine:
  1. Relative Stärke  : Performance über 1, 3 und 6 Monate im Vergleich zum Index
  2. Trendstruktur    : Kurs über steigenden 20er/50er Durchschnitten
  3. Hochs + Volumen  : Nähe zum 52-Wochen-Hoch und erhöhtes Handelsvolumen

Ergebnis:
  results/watchlist.md         -> gut lesbare Watchlist (auch am Handy)
  results/watchlist.csv        -> alle bewerteten Aktien (für die Detailanalyse)
  results/history/DATUM.csv    -> Archiv jeder Woche

Hinweis: Kein Anlagerat. Das Tool zeigt nur Kennzahlen, keine Garantie.
"""

import io
import os
import re
import sys
import time
from datetime import datetime, timezone

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
MIN_TURNOVER_EU = 1e6   # Mindest-Tagesumsatz in EUR
WEIGHTS = {"rs": 0.50, "trend": 0.25, "high": 0.25}  # Gewichtung im Gesamtscore

# Aktienlisten (werden jede Woche frisch von Wikipedia geholt)
SOURCES = [
    ("US", "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"),
    ("US", "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"),
    ("US", "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"),
    ("US", "https://en.wikipedia.org/wiki/Nasdaq-100"),
    ("EU", "https://en.wikipedia.org/wiki/DAX"),
    ("EU", "https://en.wikipedia.org/wiki/MDAX"),
    ("EU", "https://en.wikipedia.org/wiki/SDAX"),
    ("EU", "https://en.wikipedia.org/wiki/EURO_STOXX_50"),
]

# Vergleichsindex je Region
BENCHMARK = {"US": "SPY", "EU": "EXSA.DE"}  # S&P 500 ETF / STOXX Europe 600 ETF

HEADERS = {"User-Agent": "Mozilla/5.0 (momentum-screener; educational use)"}
OUT_DIR = "results"


# --------------------------------------------------------------------------
# 1) AKTIENLISTE LADEN
# --------------------------------------------------------------------------
def _find_col(columns, candidates):
    for col in columns:
        name = str(col).strip().lower()
        if name in candidates:
            return col
    return None


def load_universe():
    rows = []
    for region, url in SOURCES:
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
            sect_col = _find_col(t.columns, {"gics sector", "sector", "prime standard sector", "industry"})
            for _, r in t.iterrows():
                raw = re.sub(r"\[.*?\]", "", str(r[tick_col])).strip().upper()
                if not raw or raw == "NAN":
                    continue
                if region == "US":
                    ticker = raw.replace(".", "-")          # BRK.B -> BRK-B
                else:
                    ticker = raw if "." in raw else raw + ".DE"  # MDAX: AIXA -> AIXA.DE
                rows.append({
                    "ticker": ticker,
                    "region": region,
                    "name": str(r[name_col]) if name_col is not None else "",
                    "sector": str(r[sect_col]) if sect_col is not None else "",
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
def download_prices(tickers, chunk=150):
    """Lädt ca. 14 Monate Tageskurse. Gibt dict ticker -> DataFrame zurück."""
    data = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                df = yf.download(part, period="14mo", interval="1d", auto_adjust=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"[WARN] Download-Fehler, neuer Versuch ({e})")
                time.sleep(5)
        else:
            continue
        for t in part:
            try:
                sub = df[t] if len(part) > 1 else df
                sub = sub[["High", "Low", "Close", "Volume"]].dropna()
                if len(sub) >= 130:
                    data[t] = sub
            except Exception:
                pass
        print(f"[INFO] Kurse geladen: {min(i + chunk, len(tickers))}/{len(tickers)}")
    return data


# --------------------------------------------------------------------------
# 3) KENNZAHLEN BERECHNEN
# --------------------------------------------------------------------------
def ret(series, days):
    if len(series) <= days:
        return np.nan
    return series.iloc[-1] / series.iloc[-1 - days] - 1


def compute_metrics(df, bench_close):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = c.iloc[-1]

    r1, r3, r6 = ret(c, 21), ret(c, 63), ret(c, 126)
    b1, b3, b6 = ret(bench_close, 21), ret(bench_close, 63), ret(bench_close, 126)
    rs_raw = 0.2 * (r1 - b1) + 0.4 * (r3 - b3) + 0.4 * (r6 - b6)

    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean() if len(c) >= 200 else None

    checks = {
        "über SMA20": last > sma20.iloc[-1],
        "SMA20>SMA50": sma20.iloc[-1] > sma50.iloc[-1],
        "über SMA50": last > sma50.iloc[-1],
        "SMA50 steigt": sma50.iloc[-1] > sma50.iloc[-11],
    }
    if sma200 is not None and not np.isnan(sma200.iloc[-1]):
        checks["über SMA200"] = last > sma200.iloc[-1]
    trend_pts = sum(bool(x) for x in checks.values())
    trend_score = trend_pts / len(checks) * 100

    high52 = h.iloc[-252:].max()
    dist_high = last / high52 - 1                       # 0 = am Hoch, -0.10 = 10% darunter
    new_high_5d = h.iloc[-5:].max() >= high52 * 0.999   # neues 52W-Hoch in den letzten 5 Tagen
    vol50 = v.iloc[-55:-5].mean()
    vol_ratio = v.iloc[-5:].mean() / vol50 if vol50 > 0 else np.nan

    proximity = float(np.clip(1 + dist_high * 5, 0, 1))   # 20% unter Hoch -> 0
    vol_part = float(np.clip((vol_ratio if not np.isnan(vol_ratio) else 0) / 2, 0, 1))
    high_score = 70 * proximity + 30 * vol_part

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]

    return {
        "price": last,
        "ret_1m": r1, "ret_3m": r3, "ret_6m": r6,
        "rs_raw": rs_raw,
        "trend_pts": f"{trend_pts}/{len(checks)}",
        "trend_score": trend_score,
        "above_sma50": checks["über SMA50"],
        "sma50_rising": checks["SMA50 steigt"],
        "dist_high": dist_high,
        "new_high_5d": bool(new_high_5d),
        "vol_ratio": vol_ratio,
        "high_score": high_score,
        "atr_pct": atr / last,
        "sma20": sma20.iloc[-1],
        "stop_2atr": last - 2 * atr,
        "turnover": (c.iloc[-20:] * v.iloc[-20:]).mean(),
    }


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
            except Exception:
                continue
            if m["price"] < MIN_PRICE or m["turnover"] < min_turn or np.isnan(m["rs_raw"]):
                continue
            out.append({**r.to_dict(), **m})

    res = pd.DataFrame(out)
    if res.empty:
        return res
    # RS-Rang 1-99 innerhalb der Region (wie IBD Relative Strength Rating)
    res["rs_rank"] = res.groupby("region")["rs_raw"].rank(pct=True) * 98 + 1
    res["score"] = (WEIGHTS["rs"] * res["rs_rank"]
                    + WEIGHTS["trend"] * res["trend_score"]
                    + WEIGHTS["high"] * res["high_score"])
    # Pflicht für Momentum: Kurs über SMA50 und SMA50 steigend
    res["qualifies"] = res["above_sma50"] & res["sma50_rising"]
    return res.sort_values("score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 4) AUSGABE
# --------------------------------------------------------------------------
def pct(x):
    return "" if pd.isna(x) else f"{x * 100:+.1f}%"


def md_table(df):
    lines = ["| # | Ticker | Name | Score | RS | 1M | 3M | 6M | vom Hoch | Vol-Faktor | Trend | ATR% | Stop-Idee (2×ATR) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(df.itertuples(), 1):
        flag = " 🔥" if r.new_high_5d and r.vol_ratio >= 1.5 else ""
        lines.append(
            f"| {i} | **{r.ticker}**{flag} | {str(r.name)[:28]} | {r.score:.0f} | {r.rs_rank:.0f} | "
            f"{pct(r.ret_1m)} | {pct(r.ret_3m)} | {pct(r.ret_6m)} | {pct(r.dist_high)} | "
            f"{r.vol_ratio:.1f}× | {r.trend_pts} | {r.atr_pct * 100:.1f}% | {r.stop_2atr:.2f} |")
    return "\n".join(lines)


def write_outputs(res):
    os.makedirs(f"{OUT_DIR}/history", exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    res.to_csv(f"{OUT_DIR}/watchlist.csv", index=False, float_format="%.4f")
    res.to_csv(f"{OUT_DIR}/history/{today}.csv", index=False, float_format="%.4f")

    q = res[res.qualifies]
    us = q[q.region == "US"].head(TOP_US)
    eu = q[q.region == "EU"].head(TOP_EU)

    md = f"""# 📈 Momentum-Watchlist ({today})

**Bewertete Aktien:** {len(res)} · **Erfüllen Trendfilter:** {len(q)}

## 🇺🇸 USA (Top {TOP_US})
{md_table(us)}

## 🇪🇺 Europa (Top {TOP_EU})
{md_table(eu)}

## Legende
- **Score:** Gesamtwert 0 bis 100 (50% Relative Stärke, 25% Trend, 25% Hoch + Volumen)
- **RS:** Relative Stärke 1 bis 99 (99 = stärker als 99% der Aktien der Region)
- **vom Hoch:** Abstand zum 52-Wochen-Hoch
- **Vol-Faktor:** Volumen der letzten 5 Tage vs. Durchschnitt (über 1,5× = auffällig)
- **Trend:** erfüllte Trendkriterien (über SMA20/50/200, SMA20 > SMA50, SMA50 steigt)
- **ATR%:** typische Tagesschwankung, hilft bei Stop und Positionsgröße
- **Stop-Idee:** Kurs minus 2× ATR (nur Orientierung)
- 🔥 = neues 52-Wochen-Hoch mit hohem Volumen in den letzten 5 Tagen

> Kein Anlagerat. Kennzahlen aus der Vergangenheit garantieren keine künftige Entwicklung.
"""
    with open(f"{OUT_DIR}/watchlist.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[INFO] Watchlist geschrieben: {len(us)} US, {len(eu)} EU")


def main():
    uni = load_universe()
    if uni.empty:
        sys.exit("[ERROR] Keine Aktienliste geladen.")
    tickers = list(uni.ticker) + list(BENCHMARK.values())
    prices = download_prices(tickers)
    res = score(uni, prices)
    if res.empty:
        sys.exit("[ERROR] Keine Aktie bewertet (Kursdaten fehlen?).")
    write_outputs(res)


if __name__ == "__main__":
    main()
