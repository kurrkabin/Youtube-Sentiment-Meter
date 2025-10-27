from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# in your project env / venv

import pandas as pd
import datetime as dt
@@ -14,13 +13,6 @@
# ---- tolerant JSON repair/parse for model replies ----
import re, json

# Try yfinance for ETH price
try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False


def _safe_json_loads(raw: str) -> dict:
"""Parse 'almost JSON' from the model by fixing common issues."""
@@ -200,34 +192,40 @@ def fetch_range_for_months(channel: str, api_key: str, start_y: int, start_m: in
return pd.concat(frames, ignore_index=True)

# =========================
# =========================
# Ethereum price (CoinGecko first, Yahoo fallback — no extra deps)
# =========================
import urllib.request, json as _json

def _to_epoch_utc(d: dt.datetime) -> int:
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp())

@st.cache_data(show_spinner=True)
def fetch_eth_monthly(start_y: int, start_m: int, end_y: int, end_m: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns: YearMonth (YYYY-MM), Close (float).
    Tries CoinGecko's market_chart/range first (no key), then falls back to Yahoo CSV.
    """
    start_dt, _ = month_range(start_y, start_m)
    _, end_exclusive = month_range(end_y, end_m)  # first day AFTER end month

    # Helper to downsample daily/irregular to month-end close
    def _to_monthly_close(df: pd.DataFrame, date_col: str, price_col: str) -> pd.DataFrame:
        if df.empty:
            return df
        out = (df.set_index(date_col)[price_col]
               .resample("M").last()
               .to_frame()
               .reset_index())
        out["YearMonth"] = out[date_col].dt.strftime("%Y-%m")
        out = out[["YearMonth", price_col]].rename(columns={price_col: "Close"})
        ym_set = {f"{y}-{m:02d}" for (y, m) in months_between(start_y, start_m, end_y, end_m)}
        return out[out["YearMonth"].isin(ym_set)].reset_index(drop=True)

    # ---------- Try CoinGecko (no API key) ----------
    try:
        url = (
            "https://api.coingecko.com/api/v3/coins/ethereum/market_chart/range"
            f"?vs_currency=usd&from={_to_epoch_utc(start_dt)}&to={_to_epoch_utc(end_exclusive)}"
        )
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        prices = data.get("prices", [])  # list of [ms, price]
        if prices:
            df = pd.DataFrame(prices, columns=["ts", "px"])
            df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
            df = df[["Date", "px"]].rename(columns={"px": "Close"})
            monthly = _to_monthly_close(df, "Date", "Close")
            if not monthly.empty:
                return monthly
    except Exception:
        pass  # fall back to Yahoo

    # ---------- Fallback: Yahoo CSV ----------
    try:
        p1 = _to_epoch_utc(start_dt)
        p2 = _to_epoch_utc(end_exclusive)
        yurl = (
            "https://query1.finance.yahoo.com/v7/finance/download/ETH-USD"
            f"?period1={p1}&period2={p2}&interval=1d&events=history&includeAdjustedClose=true"
        )
        ydf = pd.read_csv(yurl, parse_dates=["Date"])
        if not ydf.empty:
            return _to_monthly_close(ydf[["Date", "Close"]], "Date", "Close")
    except Exception:
        pass

    # If both sources fail, return empty; caller will render sentiment-only.
    return pd.DataFrame(columns=["YearMonth", "Close"])

# =========================
# Classifier (model-only)
@@ -415,7 +413,6 @@ def render_overall_score(score: float):
unsafe_allow_html=True,
)


def show_sentiment_meter(score: float):
score = max(-1.0, min(1.0, float(score)))   # clamp
pos   = score + 1.0                         # map [-1,1] -> [0,2]
@@ -541,7 +538,7 @@ def render_big_score(score: float):
.sort_values("YearMonth")
)

    # --- ETH price (monthly) from Yahoo Finance
    # --- ETH price (monthly) from Yahoo Finance CSV
eth_monthly = fetch_eth_monthly(int(start_year), int(start_month), int(end_year), int(end_month))

c1, c2 = st.columns([1, 1])
@@ -566,11 +563,9 @@ def render_big_score(score: float):
with c2:
st.subheader("Monthly Sentiment Index (line) + ETH Price")

        # count distinct months in the selected range
n_months = monthly_index["YearMonth"].nunique()

if n_months > 1:
            # Base sentiment line (left axis)
sent_line = (
alt.Chart(monthly_index)
.mark_line(point=True)
@@ -584,8 +579,7 @@ def render_big_score(score: float):
.properties(height=320)
)

            if _HAS_YF and not eth_monthly.empty:
                # ETH line (right axis)
            if not eth_monthly.empty:
eth_line = (
alt.Chart(eth_monthly)
.mark_line(point=True)
@@ -597,12 +591,10 @@ def render_big_score(score: float):
)
.properties(height=320)
)

layered = alt.layer(sent_line, eth_line).resolve_scale(y='independent')
st.altair_chart(layered, use_container_width=True)
else:
                if not _HAS_YF:
                    st.info("Install `yfinance` to overlay ETH price (pip install yfinance). Showing sentiment only.")
                st.info("Couldn't fetch ETH-USD prices right now. Showing sentiment only.")
st.altair_chart(sent_line, use_container_width=True)
else:
st.caption("Monthly line chart hidden (only one month in range).")
