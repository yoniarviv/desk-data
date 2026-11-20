"""Builds data/latest.csv and data/screen.json for the paper-trading desk.
Universe: S&P 500 + Nasdaq 100 (+ extra_tickers.txt). Runs on GitHub Actions after US close.
Strategy screen: close > SMA200, SMA200 rising over 20d, ROC63 >= 15% in last 60d and reset to -5..+5%.
"""
import io, json, datetime as dt, pathlib, urllib.request
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}


def wiki_tables(url):
    html = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
    return pd.read_html(io.BytesIO(html))


def universe():
    t = set()
    sp = wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    t |= set(sp["Symbol"].astype(str))
    for tb in wiki_tables("https://en.wikipedia.org/wiki/Nasdaq-100"):
        if "Ticker" in tb.columns:
            t |= set(tb["Ticker"].astype(str))
            break
    extra = ROOT / "extra_tickers.txt"
    if extra.exists():
        t |= {x.strip().upper() for x in extra.read_text().split() if x.strip()}
    return sorted(x.replace(".", "-") for x in t)


def metrics(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    if len(c.dropna()) < 260:
        return None
    sma50, sma200 = c.rolling(50).mean(), c.rolling(200).mean()
    roc63 = c.pct_change(63) * 100
    roc20 = c.pct_change(20) * 100
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    close = float(c.iloc[-1])
    s200 = float(sma200.iloc[-1])
    slope = (s200 / float(sma200.iloc[-21]) - 1) * 100
    peak = float(roc63.iloc[-60:].max())
    r63 = float(roc63.iloc[-1])
    adv = float((c * v).rolling(50).mean().iloc[-1])
    above = close > s200
    rising = slope > 0
    reset = peak >= 15 and -5 <= r63 <= 5
    return dict(
        close=round(close, 2), sma50=round(float(sma50.iloc[-1]), 2), sma200=round(s200, 2),
        pct_vs_sma200=round((close / s200 - 1) * 100, 2), sma200_slope_20d_pct=round(slope, 2),
        roc63=round(r63, 2), roc63_peak_60d=round(peak, 2), roc20=round(float(roc20.iloc[-1]), 2),
        atr14=round(float(atr14.iloc[-1]), 2), low_20d=round(float(l.iloc[-20:].min()), 2),
        high_52w=round(float(h.iloc[-252:].max()), 2), avg_dollar_vol=round(adv),
        above_sma200=bool(above), sma200_rising=bool(rising), roc_reset=bool(reset),
        **{"pass": bool(above and rising and reset)},
    )


def main():
    tickers = universe()
    raw = yf.download(tickers, period="2y", interval="1d", auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    rows = []
    for t in tickers:
        try:
            m = metrics(raw[t].dropna(subset=["Close"]))
        except Exception:
            m = None
        if m:
            rows.append({"ticker": t, **m})
    df = pd.DataFrame(rows)
    data_date = str(raw.index[-1].date())
    df.insert(1, "data_date", data_date)
    df.to_csv(DATA / "latest.csv", index=False)

    reg = yf.download(["SPY", "QQQ", "^VIX"], period="2y", interval="1d", auto_adjust=True,
                      group_by="ticker", progress=False)
    regime = {}
    for s in ["SPY", "QQQ", "^VIX"]:
        c = reg[s]["Close"].dropna()
        regime[s] = dict(close=round(float(c.iloc[-1]), 2), sma200=round(float(c.rolling(200).mean().iloc[-1]), 2),
                         above_sma200=bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1]),
                         roc20=round(float(c.pct_change(20).iloc[-1] * 100), 2))
    passing = df[df["pass"]].sort_values("avg_dollar_vol", ascending=False)
    near = df[(df.above_sma200) & (df.sma200_rising) & (~df["pass"]) & (df.roc63_peak_60d >= 15)
              & (df.roc63.between(-8, 8))]
    out = dict(
        generated_utc=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z", data_date=data_date,
        universe_size=len(df), regime=regime,
        breadth=dict(pct_above_sma200=round(float(df.above_sma200.mean() * 100), 1),
                     pct_above_sma50=round(float((df.close > df.sma50).mean() * 100), 1)),
        passing=passing.to_dict("records"), near_misses=near.to_dict("records"),
    )
    (DATA / "screen.json").write_text(json.dumps(out, indent=1))
    print(f"{data_date}: {len(df)} tickers, {len(passing)} pass, {len(near)} near")


if __name__ == "__main__":
    main()
