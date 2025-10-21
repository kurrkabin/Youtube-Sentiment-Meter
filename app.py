import streamlit as st
from openai import OpenAI
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pandas as pd
import datetime as dt
import isodate
import calendar, io, json, math, re
import altair as alt
from typing import List, Tuple

# ---- load secrets ONCE ----
OPENAI_KEY  = st.secrets.get("openai", {}).get("api_key", "")
YOUTUBE_KEY = st.secrets.get("google", {}).get("youtube_api_key", "")

if not OPENAI_KEY:
    st.warning("Missing OpenAI key in Settings → Secrets. Expected:\n[openai]\napi_key = \"...\"")
if not YOUTUBE_KEY:
    st.warning("Missing YouTube key in Settings → Secrets. Expected:\n[google]\nyoutube_api_key = \"...\"")

# OpenAI client
client = OpenAI(api_key=OPENAI_KEY)



# -----------------------
# Page setup
# -----------------------
st.set_page_config(page_title="YouTube Sentiment Scanner (GPT)", layout="wide")

st.title("YouTube Sentiment Scanner — Bullish / Bearish / Neutral (GPT)")

st.markdown(
    "Type a channel (e.g., `@DataDash`), pick a month, and get a sentiment index (−1..+1) "
    "based on video **titles** classified with GPT."
)



# -----------------------
# Helpers
# -----------------------
def month_range(year: int, month: int) -> Tuple[dt.datetime, dt.datetime]:
    start = dt.datetime(year, month, 1)
    end = dt.datetime(year + 1, 1, 1) if month == 12 else dt.datetime(year, month + 1, 1)
    return start, end

def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

# -----------------------
# YouTube fetch
# -----------------------
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

# -----------------------
# GPT classification (prompt + few-shot + keyword backstop)
# -----------------------
OPENAI_JSON_INSTRUCTIONS = (
    "You are a STRICT market-sentiment classifier for crypto/finance video titles.\n"
    "Output exactly ONE label per title in {Bullish, Bearish, Neutral}.\n"
    "Definitions:\n"
    "- Bullish: clear upside phrasing (surge, rally, pump, breakout, soar, moon, ATH, rip, rebound, bounce back).\n"
    "- Bearish: clear downside phrasing (drop, dump, crash, plunge, liquidation, breakdown, rejection, danger, warning, damage).\n"
    "- Neutral: tutorials/news/sponsors or hedged questions with NO clear directional bias.\n"
    "Edge rules:\n"
    "- Strong down/up verbs trump question marks: 'Drops 30%?' is Bearish; 'Short squeeze incoming?' is Bullish.\n"
    "- If mixed/ambiguous, choose Neutral.\n"
    "Return ONLY JSON: {\"labels\":[{\"index\":0,\"label\":\"Bullish\"}]}\n"
)
FEW_SHOT = [
    {"role": "user", "content": json.dumps([{"index":0,"title":"Crypto Damage Report"},{"index":1,"title":"Ethereum Drops 30% - Now What?"}])},
    {"role": "assistant", "content": json.dumps({"labels":[{"index":0,"label":"Bearish"},{"index":1,"label":"Bearish"}]})},
    {"role": "user", "content": json.dumps([{"index":0,"title":"Bitcoin Bounces Back After Friday’s Dump | What This Means"},{"index":1,"title":"Is the Top In for Bitcoin?"}])},
    {"role": "assistant", "content": json.dumps({"labels":[{"index":0,"label":"Bullish"},{"index":1,"label":"Neutral"}]})},
]

_BULL_KWS = set([
    "surge","surges","soar","soars","rally","rallies","pump","pumps","moon","breakout","spike","spikes",
    "ath","new high","record high","rip","rips","rebound","rebounds","bounce","bounces","recovery","recover",
    "short squeeze","squeeze incoming","wrecked","wrong about this rally","supercycle","on the cusp","cusp of","ready to run","set to run","altseason","altcoin season"
])
_BEAR_KWS = set([
    "drop","drops","dump","dumps","crash","crashes","plunge","plunges","collapse","liquidation","liquidations",
    "rekt","bloodbath","bearish","bear market","sell-off","sell off","breakdown","rejection","danger","warning","damage","damage report","downtrend","bull trap","trap"
])

def _extract_json_block(s: str):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        l, r = s.find("{"), s.rfind("}")
        cand = s[l:r+1] if l>=0 and r>l else ""
        try:
            return json.loads(cand) if cand else {}
        except Exception:
            return {}

def _kw_override(title: str, model_label: str) -> str:
    t = title.lower()
    bull = any(k in t for k in _BULL_KWS)
    bear = any(k in t for k in _BEAR_KWS)
    if bear and not bull:
        return "Bearish"
    if bull and not bear:
        return "Bullish"
    return model_label

@st.cache_data(show_spinner=True)
def classify_titles_chatgpt(titles: List[str], api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    if not api_key:
        raise RuntimeError("Missing OpenAI API key in secrets.")
    client = OpenAI(api_key=api_key)

    out = []
    BATCH = 20
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i+BATCH]
        payload = [{"index": j, "title": t} for j, t in enumerate(chunk)]
        user_prompt = OPENAI_JSON_INSTRUCTIONS + json.dumps(payload, ensure_ascii=False)
        needed_tokens = max(300, len(chunk) * 16)
        try:
            resp = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=needed_tokens,
                messages=[
                    {"role":"system","content":"You output strict JSON only."},
                    *FEW_SHOT,
                    {"role":"user","content": user_prompt}
                ],
            )
            raw = resp.choices[0].message.content or ""
            data = raw if isinstance(raw, dict) else json.loads(raw)
        except Exception:
            # fallback when JSON mode is ignored
            resp = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=needed_tokens,
                messages=[
                    {"role":"system","content":"You output strict JSON only."},
                    *FEW_SHOT,
                    {"role":"user","content": user_prompt}
                ],
            )
            data = _extract_json_block(resp.choices[0].message.content or "")

        labels = ["Neutral"] * len(chunk)
        for item in (data.get("labels") or []):
            idx = item.get("index"); lab = item.get("label")
            if isinstance(idx, int) and 0 <= idx < len(chunk) and lab in {"Bullish","Bearish","Neutral"}:
                labels[idx] = lab

        # keyword backstop
        labels = [_kw_override(chunk[j], labels[j]) for j in range(len(chunk))]
        out.extend(labels)

    if len(out) != len(titles):
        out += ["Neutral"] * (len(titles) - len(out))
    return out

def compute_index(df: pd.DataFrame, labels: List[str]) -> Tuple[float, pd.DataFrame]:
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    df["Sentiment"] = labels
    df["SentimentScore"] = df["Sentiment"].map(score_map)
    idx_val = float(df["SentimentScore"].mean()) if len(df) else 0.0
    counts = pd.Series(labels).value_counts().reindex(["Bullish","Bearish","Neutral"]).fillna(0).astype(int)
    counts_tbl = pd.DataFrame({"Category": counts.index, "Count (1-last)": counts.values})
    return idx_val, counts_tbl

# -----------------------
# UI
# -----------------------
with st.sidebar:
    st.header("Inputs")

    channel = st.text_input("Channel handle", value="@DataDash")

    now = dt.datetime.utcnow()
    month_names = list(calendar.month_name)[1:]  # ["January", ... , "December"]
    month_name  = st.selectbox("Month", month_names, index=now.month - 1)
    month       = month_names.index(month_name) + 1

    year = st.number_input("Year", min_value=2015, max_value=2035, value=now.year, step=1)

    min_minutes = st.number_input(
        "Min video minutes", min_value=1, max_value=120, value=5, step=1,
        help="Videos shorter than this are excluded."
    )

    # Fixed model for this app; change the list if you want to allow others.
    model = st.selectbox("OpenAI model", ["gpt-4o-mini"], index=0, disabled=True,
                         help="Fixed to gpt-4o-mini for this app.")
    run = st.button("Run")


def sentiment_zone(score: float) -> str:
    if score > 0.4:
        return "Greed"
    if score < 0.0:
        return "Fear"
    return "Neutral"

def show_sentiment_meter(score: float):
    """
    A clean horizontal meter with red / gray / green zones and a needle.
    Score is in [-1, +1]. Zones:
      - [-1, 0): Fear (red)
      - [0, 0.4]: Neutral (gray)
      - (0.4, 1]: Greed (green)
    """
    score = max(-1.0, min(1.0, float(score)))       # clamp
    pos   = score + 1.0                              # map [-1,1] -> [0,2]

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
    })).mark_text(dy=-10, fontWeight="bold").encode(x="pos:Q", text="txt:N")

    st.altair_chart((base + needle + label).properties(width=520), use_container_width=False)

# -----------------------
# Run
# -----------------------
if run:
    try:
        df = fetch_channel_videos_for_month(
            channel, YOUTUBE_KEY, int(year), int(month), int(min_minutes) * 60
        )
    except Exception as e:
        st.error(f"YouTube fetch failed: {e}")
        st.stop()

    if df.empty:
        st.warning("No (>= min minutes) videos found for that month.")
        st.stop()

    st.success(f"Found {len(df)} videos.")
    st.write("Classifying with ChatGPT…")
    try:
        labels = classify_titles_chatgpt(
            df["Title"].astype(str).tolist(), OPENAI_KEY, model_name=model
        )
    except Exception as e:
        st.error(f"ChatGPT classification failed: {e}")
        st.stop()

    idx_val, counts_tbl = compute_index(df, labels)

    # Sentiment meter
    st.subheader("Sentiment Index (−1..+1)")
    show_sentiment_meter(idx_val)

    # Topline table (no left index)
    topline = pd.DataFrame([
        ["Total Videos", int(len(df))],
        ["Total Views", int(df["Views"].sum()) if len(df) else 0],
        ["Avg Views/Video", round(float(df["Views"].mean()), 2) if len(df) else 0.0],
        ["Sentiment Index (−1..+1)", round(idx_val, 3)],
    ], columns=["Metric", "Value"])

    # Lay out the two tables side by side
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Topline")
        st.dataframe(topline, use_container_width=True, hide_index=True)

        st.subheader("Category counts (1-last)")
        st.dataframe(counts_tbl, use_container_width=True, hide_index=True)

    with c2:
        prev = df.drop(columns=["VideoId"], errors="ignore").copy()
        prev.insert(0, "No.", range(1, len(prev) + 1))
        st.subheader("Labeled Videos")
        st.dataframe(prev, use_container_width=True, height=520, hide_index=True)

    # Downloads
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV", data=csv_buf.getvalue(),
        file_name=f"{channel.strip('@').lower()}_{year}-{int(month):02d}.csv",
        mime="text/csv"
    )

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Videos")
        topline.to_excel(xw, index=False, sheet_name="Topline")
        counts_tbl.to_excel(xw, index=False, sheet_name="SentimentCounts")
    st.download_button(
        "Download Excel", data=xlsx_buf.getvalue(),
        file_name=f"{channel.strip('@').lower()}_{year}-{int(month):02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Videos")
        topline.to_excel(xw, index=False, sheet_name="Topline")
        counts_tbl.to_excel(xw, index=False, sheet_name="SentimentCounts")
    st.download_button("Download Excel", data=xlsx_buf.getvalue(), file_name=f"{channel.strip('@').lower()}_{year}-{int(month):02d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.sidebar.button("Clear cache (dev)"):
    st.cache_data.clear()
    st.rerun()

