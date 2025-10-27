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
