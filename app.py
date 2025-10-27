import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
# in your project env / venv

import pandas as pd
import datetime as dt
import isodate
import calendar, io, json, math, re
import altair as alt
from typing import List, Tuple

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
    if not raw:
        return {}
    s = raw.strip()

    # remove ```json fences
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s,
                   flags=re.IGNORECASE | re.DOTALL).strip()

    # try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # extract largest {...} block
    l, r = s.find("{"), s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]

    # fix .85 -> 0.85
    s = re.sub(r'(:\s*)\.(\d+)', r'\g<1>0.\2', s)
    s = re.sub(r'(\[|\{)\s*\.(\d+)', r'\g<1>0.\2', s)

    # remove trailing commas before } or ]
    s = re.sub(r',(\s*[}\]])', r'\1', s)

    try:
        return json.loads(s)
    except Exception:
        return {}

# =========================
# Secrets / Clients
# =========================
OPENAI_KEY  = st.secrets.get("openai", {}).get("api_key", "")
YOUTUBE_KEY = st.secrets.get("google", {}).get("youtube_api_key", "")

if not OPENAI_KEY:
    st.warning("Missing OpenAI key in Settings → Secrets. Expected:\n[openai]\napi_key = \"...\"")
if not YOUTUBE_KEY:
    st.warning("Missing YouTube key in Settings → Secrets. Expected:\n[google]\nyoutube_api_key = \"...\"")

# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="YouTube Sentiment (GPT)", layout="wide")
st.title("YouTube Fear & Greed Index")
st.markdown(
    "Select a channel (e.g., `@DataDash`), pick a **start** and **end** month, "
    "and get an overall **Fear & Greed Index** + per-month breakdown based on video **titles** classified with AI."
)

# =========================
# Helpers
# =========================
def month_range(year: int, month: int) -> Tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, month, 1)
    end = dt.datetime(year + 1, 1, 1) if month == 12 else dt.datetime(year, month + 1, 1)
    return start, end

def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

def months_between(y1: int, m1: int, y2: int, m2: int):
    """Yield (year, month) inclusive from (y1,m1) to (y2,m2)."""
    a = y1 * 12 + (m1 - 1)
    b = y2 * 12 + (m2 - 1)
    for idx in range(a, b + 1):
        y = idx // 12
        m = idx % 12 + 1
        yield y, m

# =========================
# YouTube fetch
# =========================
@st.cache_data(show_spinner=True)
def fetch_channel_videos_for_month(channel_handle: str, api_key: str, year: int, month: int, min_seconds: int):
    if not api_key:
        raise RuntimeError("Missing YouTube API key in secrets.")
    if not channel_handle.startswith("@"):
        channel_handle = "@" + channel_handle

    try:
        yt = build("youtube", "v3", developerKey=api_key)
        ch = yt.channels().list(part="id,contentDetails", forHandle=channel_handle.lstrip("@")).execute()
    except HttpError as e:
        if e.resp.status == 403 and "accessNotConfigured" in str(e):
            raise RuntimeError("YouTube Data API v3 is not enabled for this key’s project.")
        raise

    items = ch.get("items", [])
    if not items:
        raise RuntimeError(f"Channel not found: {channel_handle}")
    uploads = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

    start, end = month_range(year, month)
    raw, token = [], None

    while True:
        res = yt.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads, maxResults=50, pageToken=token
        ).execute()
        for it in res.get("items", []):
            sn = it.get("snippet", {})
            pub_raw = sn.get("publishedAt")
            if not pub_raw:
                continue
            pub = dt.datetime.fromisoformat(pub_raw.replace("Z", "+00:00")).replace(tzinfo=None)
            if start <= pub < end:
                vid = sn["resourceId"]["videoId"]
                title = sn.get("title", "")
                url = f"https://www.youtube.com/watch?v={vid}"
                raw.append({"Datetime": pub, "Title": title, "URL": url, "VideoId": vid})
        token = res.get("nextPageToken")
        if not token:
            break

    if not raw:
        return pd.DataFrame(columns=["Datetime","Title","URL","VideoId","DurationSec","Views"])

    # durations + views
    durations, views = {}, {}
    for i in range(0, len(raw), 50):
        ids = [r["VideoId"] for r in raw[i:i+50]]
        vres = yt.videos().list(part="contentDetails,statistics", id=",".join(ids)).execute()
        for item in vres.get("items", []):
            vid = item["id"]
            dur_iso = item.get("contentDetails", {}).get("duration")
            if dur_iso:
                try:
                    durations[vid] = int(isodate.parse_duration(dur_iso).total_seconds())
                except Exception:
                    pass
            vc = item.get("statistics", {}).get("viewCount")
            try:
                views[vid] = int(vc) if vc is not None else 0
            except Exception:
                views[vid] = 0

    rows = []
    for r in raw:
        dsec = durations.get(r["VideoId"])
        if dsec is None or (isinstance(dsec, float) and math.isnan(dsec)) or dsec < min_seconds:
            continue
        rows.append([
            r["Datetime"].isoformat(), r["Title"], r["URL"], r["VideoId"], int(dsec), int(views.get(r["VideoId"], 0) or 0)
        ])

    df = pd.DataFrame(rows, columns=["Datetime","Title","URL","VideoId","DurationSec","Views"])
    if df.empty:
        return df
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["__norm"] = df["Title"].apply(_norm_title)
    df = df.drop_duplicates(subset="__norm", keep="first").drop(columns="__norm")
    return df

@st.cache_data(show_spinner=True)
def fetch_range_for_months(channel: str, api_key: str, start_y: int, start_m: int, end_y: int, end_m: int, min_seconds: int):
    """Loop months, fetch, and concatenate. Soft-fails individual months."""
    frames = []
    for (yy, mm) in months_between(start_y, start_m, end_y, end_m):
        try:
            dfm = fetch_channel_videos_for_month(channel, api_key, yy, mm, min_seconds)
            if not dfm.empty:
                dfm["Year"] = yy
                dfm["Month"] = mm
                frames.append(dfm)
        except Exception as e:
            st.info(f"Skipping {yy}-{mm:02d}: {e}")
    if not frames:
        return pd.DataFrame(columns=["Datetime","Title","URL","VideoId","DurationSec","Views","Year","Month"])
    return pd.concat(frames, ignore_index=True)

# =========================
# Ethereum price (Yahoo Finance)
# =========================
@st.cache_data(show_spinner=True)
def fetch_eth_monthly(start_y: int, start_m: int, end_y: int, end_m: int) -> pd.DataFrame:
    """
    Fetch ETH-USD daily OHLC from Yahoo Finance and downsample to month-end Close.
    Returns DataFrame with columns: YearMonth (YYYY-MM), Close (float).
    """
    if not _HAS_YF:
        # Return empty; caller will handle gracefully
        return pd.DataFrame(columns=["YearMonth", "Close"])

    start_dt, _ = month_range(start_y, start_m)
    _, end_exclusive = month_range(end_y, end_m)  # end is first day of the month AFTER end
    # yfinance end is exclusive; OK as is
    df = yf.download("ETH-USD", start=start_dt.date(), end=end_exclusive.date(), progress=False)
    if df is None or df.empty:
        return pd.DataFrame(columns=["YearMonth", "Close"])

    # Resample to month end (last business day close in month)
    monthly = df["Close"].resample("M").last().to_frame().reset_index()
    monthly["YearMonth"] = monthly["Date"].dt.strftime("%Y-%m")
    monthly = monthly[["YearMonth", "Close"]]

    # Keep only the selected months (defensive)
    ym_set = {f"{y}-{m:02d}" for (y, m) in months_between(start_y, start_m, end_y, end_m)}
    monthly = monthly[monthly["YearMonth"].isin(ym_set)].reset_index(drop=True)
    return monthly

# =========================
# Classifier (model-only)
# =========================
def classify_titles_chatgpt(
    titles: List[str],
    api_key: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    neutral_floor: float = 0.55,
    margin_floor: float = 0.10
):
    if not api_key:
        raise RuntimeError("Missing OpenAI API key in secrets.")

    client = OpenAI(api_key=api_key)

    SYS = (
        "You are a STRICT sentiment classifier for crypto/finance video titles.\n"
        "Assign exactly one label: Bullish, Bearish, or Neutral.\n\n"
        "Definitions:\n"
        "- Bullish: clear up-move language or upside expectation.\n"
        "- Bearish: clear down-move language or downside expectation.\n"
        "- Neutral: tutorials, ads, generic news, or mixed/hedged titles (e.g., 'X soars but Y crashes').\n\n"
        "Rules:\n"
        "1) Mixed/opposing signals → Neutral.\n"
        "2) Question marks / 'might/could' → Neutral unless clearly one-sided.\n"
        "3) Clickbait adjectives/emojis do not decide direction.\n"
        "4) The 🚀 emoji usually signals Bullish unless the rest of the title clearly indicates a mixed or negative situation.\n\n"
        "Output JSON ONLY:\n"
        "{\"items\":[{\"index\":0, \"label\":\"Bullish|Bearish|Neutral\", "
        "\"scores\":{\"Bullish\":0.0,\"Bearish\":0.0,\"Neutral\":0.0}}]}\n"
        "Scores should sum ~1; avoid 1.00 unless truly certain."
)

    FEW_SHOT = [
        {
            "role":"user",
            "content": json.dumps([
                {"index":0,"title":"Bitcoin soars but altcoins crash hard"},
                {"index":1,"title":"Is this the final Bitcoin top?"},
                {"index":2,"title":"Ethereum down 12% after liquidation cascade"}
            ], ensure_ascii=False)
        },
        {
            "role":"assistant",
            "content": json.dumps({
                "items":[
                    {"index":0,"label":"Neutral","scores":{"Bullish":0.35,"Bearish":0.35,"Neutral":0.30}},
                    {"index":1,"label":"Neutral","scores":{"Bullish":0.30,"Bearish":0.25,"Neutral":0.45}},
                    {"index":2,"label":"Bearish","scores":{"Bullish":0.05,"Bearish":0.85,"Neutral":0.10}}
                ]
            })
        }
    ]

    out_labels = []
    BATCH = 20
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i+BATCH]
        payload = [{"index": j+i, "title": t} for j, t in enumerate(chunk)]
        user_prompt = json.dumps(payload, ensure_ascii=False)

        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            response_format={"type":"json_object"},
            messages=[
                {"role":"system","content": SYS},
                *FEW_SHOT,
                {"role":"user","content": user_prompt}
            ],
            max_tokens=800
        )

        raw = resp.choices[0].message.content or "{}"
        data = _safe_json_loads(raw)  # <--- robust parse

        by_index = {it.get("index"): it for it in (data.get("items") or [])}
        for j, _ in enumerate(chunk):
            item = by_index.get(j+i, {})
            scrs = (item.get("scores") or {})
            b = float(scrs.get("Bullish", 0.0))
            d = float(scrs.get("Bearish", 0.0))
            n = float(scrs.get("Neutral", 0.0))

            if (b + d + n) <= 0:
                out_labels.append(item.get("label") or "Neutral")
                continue

            scores = {"Bullish": b, "Bearish": d, "Neutral": n}
            top_label = max(scores, key=scores.get)
            top = scores[top_label]
            second = max(v for k, v in scores.items() if k != top_label)

            if top < neutral_floor or (top - second) < margin_floor:
                out_labels.append("Neutral")
            else:
                out_labels.append(top_label)

    if len(out_labels) != len(titles):
        out_labels += ["Neutral"] * (len(titles) - len(out_labels))

    return out_labels


# =========================
# Compute index
# =========================
def compute_index(df: pd.DataFrame, labels: List[str]) -> Tuple[float, pd.DataFrame]:
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    df["Sentiment"] = labels
    df["SentimentScore"] = df["Sentiment"].map(score_map)
    idx_val = float(df["SentimentScore"].mean()) if len(df) else 0.0
    counts = pd.Series(labels).value_counts().reindex(["Bullish","Bearish","Neutral"]).fillna(0).astype(int)
    counts_tbl = pd.DataFrame({"Category": counts.index, "Count (1-last)": counts.values})
    return idx_val, counts_tbl

# =========================
# Sidebar (range inputs)
# =========================
with st.sidebar:
    st.header("Inputs")

    channel = st.text_input("Channel handle", value="@DataDash")

    month_names = list(calendar.month_name)[1:]  # ["January", ..., "December"]
    now = dt.datetime.utcnow()

    st.subheader("Start")
    start_month_name = st.selectbox("Start month", month_names, index=max(0, now.month-1), key="start_m")
    start_year = st.number_input("Start year", min_value=2015, max_value=2035, value=now.year, step=1, key="start_y")

    st.subheader("End")
    end_month_name = st.selectbox("End month", month_names, index=max(0, now.month-1), key="end_m")
    end_year = st.number_input("End year", min_value=2015, max_value=2035, value=now.year, step=1, key="end_y")

    start_month = month_names.index(start_month_name) + 1
    end_month   = month_names.index(end_month_name) + 1

    min_minutes = st.number_input(
        "Min video minutes", min_value=1, max_value=120, value=5, step=1,
        help="Videos shorter than this are excluded."
    )

    model = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="mini = cheap/fast | 4o = smarter but pricier"
    )

    run = st.button("Run")

# =========================
# Sentiment meter
# =========================
def sentiment_zone(score: float) -> str:
    if score > 0.4:
        return "Greed"
    if score < 0.0:
        return "Fear"
    return "Neutral"

def render_overall_score(score: float):
    """
    Big, color-coded overall score.
    Greed (> 0.40) = green, Fear (< 0) = red, otherwise black Neutral.
    """
    label = sentiment_zone(score)
    color = "#2ca02c" if score > 0.40 else ("#d62728" if score < 0 else "#111111")
    st.markdown(
        f"""
        <div style="
            font-size: 88px;               /* was 56px */
            font-weight: 900;
            line-height: 1.02;
            letter-spacing: -1px;
            margin: 6px 0 18px 0;
            color: {color};
        ">
            {score:+.2f} • {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_sentiment_meter(score: float):
    score = max(-1.0, min(1.0, float(score)))   # clamp
    pos   = score + 1.0                         # map [-1,1] -> [0,2]

    zones = pd.DataFrame([
        {"zone": "Fear",    "x0": 0.0, "x1": 1.0, "color": "#d62728"},
        {"zone": "Neutral", "x0": 1.0, "x1": 1.4, "color": "#999999"},
        {"zone": "Greed",   "x0": 1.4, "x1": 2.0, "color": "#2ca02c"},
    ])

    base = alt.Chart(zones).mark_bar(height=32).encode(
        x=alt.X("x0:Q", axis=None, scale=alt.Scale(domain=(0, 2))),
        x2="x1:Q",
        color=alt.Color("zone:N", scale=None, legend=None),
        tooltip=["zone:N"]
    )

    needle = alt.Chart(pd.DataFrame({"pos": [pos]})).mark_rule(
        color="black", size=3
    ).encode(x="pos:Q")

    label = alt.Chart(pd.DataFrame({
        "pos": [pos],
        "txt": [f"{score:+.2f} • {sentiment_zone(score)}"]
    })).mark_text(dy=-10, fontWeight="bold").encode(
        x="pos:Q", text="txt:N"
    )

    st.altair_chart((base + needle + label).properties(width=520), use_container_width=False)

def render_big_score(score: float):
    # label + color by your rules
    if score > 0.4:
        label, color = "Greed", "#2ca02c"    # green
    elif score < 0.0:
        label, color = "Fear", "#d62728"     # red
    else:
        label, color = "Neutral", "#111111"  # black

    st.markdown(
        f"""
        <div style="
            font-size:64px;
            font-weight:800;
            color:{color};
            line-height:1.05;
            margin: 4px 0 12px 0;
        ">
            {score:+.2f} • {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Run
# =========================
if run:
    # Range validation
    if (int(end_year), int(end_month)) < (int(start_year), int(start_month)):
        st.error("End must be equal to or after Start.")
        st.stop()

    # --- Fetch
    try:
        df = fetch_range_for_months(
            channel=channel,
            api_key=YOUTUBE_KEY,
            start_y=int(start_year),
            start_m=int(start_month),
            end_y=int(end_year),
            end_m=int(end_month),
            min_seconds=int(min_minutes) * 60,
        )
    except Exception as e:
        st.error(f"YouTube fetch failed: {e}")
        st.stop()

    if df.empty:
        st.warning("No (>= min minutes) videos found in that range.")
        st.stop()

    st.success(f"Found {len(df)} videos across the selected range.")
    st.write("Classifying with ChatGPT…")

    # --- Classify
    try:
        labels = classify_titles_chatgpt(
            df["Title"].astype(str).tolist(),
            OPENAI_KEY,
            model_name=model,
            temperature=0.0
        )
    except Exception as e:
        st.error(f"ChatGPT classification failed: {e}")
        st.stop()

    # --- Overall index & meter
    idx_val, counts_tbl = compute_index(df, labels)
    st.subheader("Overall Sentiment Index Score")
    render_overall_score(idx_val)

    # --- Monthly breakdown (counts)
    df["Sentiment"] = labels
    monthly_counts = (
        df.assign(YearMonth=df["Year"].astype(str) + "-" + df["Month"].astype(int).astype(str).str.zfill(2))
          .groupby("YearMonth")["Sentiment"]
          .value_counts()
          .unstack(fill_value=0)
          .reindex(columns=["Bullish","Bearish","Neutral"], fill_value=0)
          .reset_index()
          .sort_values("YearMonth")
    )

    # --- Monthly index (line chart)
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    df["SentimentScore"] = df["Sentiment"].map(score_map)
    monthly_index = (
        df.assign(YearMonth=df["Year"].astype(str) + "-" + df["Month"].astype(int).astype(str).str.zfill(2))
          .groupby("YearMonth")["SentimentScore"]
          .mean()
          .reset_index()
          .sort_values("YearMonth")
    )

    # --- ETH price (monthly) from Yahoo Finance
    eth_monthly = fetch_eth_monthly(int(start_year), int(start_month), int(end_year), int(end_month))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Topline")
        topline = pd.DataFrame(
            [
                ["Total Videos", int(len(df))],
                ["Total Views", int(df["Views"].sum()) if len(df) else 0],
                ["Avg Views/Video", round(float(df["Views"].mean()), 2) if len(df) else 0.0],
                ["Sentiment Index (−1..+1)", round(idx_val, 3)],
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(topline, use_container_width=True, hide_index=True)

        st.subheader("Category counts (1-last) — Overall")
        st.dataframe(counts_tbl, use_container_width=True, hide_index=True)

        st.subheader("Monthly breakdown (counts)")
        st.dataframe(monthly_counts, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Monthly Sentiment Index (line) + ETH Price")

        # count distinct months in the selected range
        n_months = monthly_index["YearMonth"].nunique()

        if n_months > 1:
            # Base sentiment line (left axis)
            sent_line = (
                alt.Chart(monthly_index)
                .mark_line(point=True)
                .encode(
                    x=alt.X("YearMonth:N", sort=None, title="Month"),
                    y=alt.Y("SentimentScore:Q",
                            scale=alt.Scale(domain=[-1, 1]),
                            title="Index (-1..+1)"),
                    tooltip=["YearMonth:N", "SentimentScore:Q"],
                )
                .properties(height=320)
            )

            if _HAS_YF and not eth_monthly.empty:
                # ETH line (right axis)
                eth_line = (
                    alt.Chart(eth_monthly)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("YearMonth:N", sort=None, title="Month"),
                        y=alt.Y("Close:Q",
                                axis=alt.Axis(orient="right", title="ETH-USD (Close)")),
                        tooltip=["YearMonth:N", alt.Tooltip("Close:Q", format=",.2f", title="ETH Close")],
                    )
                    .properties(height=320)
                )

                layered = alt.layer(sent_line, eth_line).resolve_scale(y='independent')
                st.altair_chart(layered, use_container_width=True)
            else:
                if not _HAS_YF:
                    st.info("Install `yfinance` to overlay ETH price (pip install yfinance). Showing sentiment only.")
                st.altair_chart(sent_line, use_container_width=True)
        else:
            st.caption("Monthly line chart hidden (only one month in range).")

        st.subheader("Labeled Videos")
        prev = df.drop(columns=["VideoId"], errors="ignore").copy()
        prev.insert(0, "No.", range(1, len(prev) + 1))
        st.dataframe(prev, use_container_width=True, height=520, hide_index=True)

    # --- Downloads (unique keys)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue(),
        file_name=f"{channel.strip('@').lower()}_{start_year}-{int(start_month):02d}_to_{end_year}-{int(end_month):02d}.csv",
        mime="text/csv",
        key="dl_csv",
    )

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Videos")
        topline.to_excel(xw, index=False, sheet_name="Topline")
        counts_tbl.to_excel(xw, index=False, sheet_name="OverallCounts")
        monthly_counts.to_excel(xw, index=False, sheet_name="MonthlyCounts")
        monthly_index.to_excel(xw, index=False, sheet_name="MonthlyIndex")
        if not eth_monthly.empty:
            eth_monthly.to_excel(xw, index=False, sheet_name="ETH_Monthly")
    st.download_button(
        "Download Excel",
        data=xlsx_buf.getvalue(),
        file_name=f"{channel.strip('@').lower()}_{start_year}-{int(start_month):02d}_to_{end_year}-{int(end_month):02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx",
    )
