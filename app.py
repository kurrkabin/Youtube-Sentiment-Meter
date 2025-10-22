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

# =========================
# Secrets & basic setup
# =========================
OPENAI_KEY  = st.secrets.get("openai", {}).get("api_key", "")
YOUTUBE_KEY = st.secrets.get("google", {}).get("youtube_api_key", "")

if not OPENAI_KEY:
    st.warning("Missing OpenAI key in Settings → Secrets. Expected:\n[openai]\napi_key = \"...\"")
if not YOUTUBE_KEY:
    st.warning("Missing YouTube key in Settings → Secrets. Expected:\n[google]\nyoutube_api_key = \"...\"")

client = OpenAI(api_key=OPENAI_KEY)

st.set_page_config(page_title="YouTube Fear & Greed Index", layout="wide")
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

def iter_months(y1: int, m1: int, y2: int, m2: int):
    """Inclusive (y1,m1) .. (y2,m2) month iterator."""
    cur = dt.date(y1, m1, 1)
    end = dt.date(y2, m2, 1)
    while cur <= end:
        yield cur.year, cur.month
        # bump month
        if cur.month == 12:
            cur = dt.date(cur.year + 1, 1, 1)
        else:
            cur = dt.date(cur.year, cur.month + 1, 1)

def _norm_title(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()

# =========================
# YouTube fetch (range)
# =========================
@st.cache_data(show_spinner=True)
def fetch_channel_videos_for_range(channel_handle: str, api_key: str,
                                   start_year: int, start_month: int,
                                   end_year: int, end_month: int,
                                   min_seconds: int) -> pd.DataFrame:
    """
    Returns a dataframe of all videos in the inclusive month range,
    with per-video duration (sec) & views and a YearMonth column.
    """
    if not api_key:
        raise RuntimeError("Missing YouTube API key in secrets.")
    if not channel_handle.startswith("@"):
        channel_handle = "@" + channel_handle

    # Channel & uploads playlist
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

    # Pull all items once, then filter by date
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
            vid = sn["resourceId"]["videoId"]
            title = sn.get("title", "")
            url = f"https://www.youtube.com/watch?v={vid}"
            raw.append({"Datetime": pub, "Title": title, "URL": url, "VideoId": vid})
        token = res.get("nextPageToken")
        if not token:
            break

    if not raw:
        return pd.DataFrame(columns=["Datetime","Title","URL","VideoId","DurationSec","Views","YearMonth"])

    # Limit to our month windows
    keep = []
    for (y, m) in iter_months(start_year, start_month, end_year, end_month):
        s, e = month_range(y, m)
        for r in raw:
            if s <= r["Datetime"] < e:
                keep.append(r)

    if not keep:
        return pd.DataFrame(columns=["Datetime","Title","URL","VideoId","DurationSec","Views","YearMonth"])

    # durations + views
    durations, views = {}, {}
    for i in range(0, len(keep), 50):
        ids = [r["VideoId"] for r in keep[i:i+50]]
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
    for r in keep:
        dsec = durations.get(r["VideoId"])
        if dsec is None or (isinstance(dsec, float) and math.isnan(dsec)) or dsec < min_seconds:
            continue
        year_month = r["Datetime"].strftime("%Y-%m")
        rows.append([
            r["Datetime"].isoformat(), r["Title"], r["URL"], r["VideoId"], int(dsec),
            int(views.get(r["VideoId"], 0) or 0), year_month
        ])

    df = pd.DataFrame(rows, columns=["Datetime","Title","URL","VideoId","DurationSec","Views","YearMonth"])
    if df.empty:
        return df
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["__norm"] = df["Title"].apply(_norm_title)
    df = df.drop_duplicates(subset="__norm", keep="first").drop(columns="__norm")
    return df

# =========================
# Robust JSON loader
# =========================
def _safe_json_loads(raw: str) -> dict:
    """
    Best-effort JSON parser for model replies.
    - Strips ```json fences
    - Extracts innermost {...}
    - Fixes .85 -> 0.85; removes trailing commas
    Returns {} on failure.
    """
    if not raw:
        return {}

    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s,
                   flags=re.IGNORECASE | re.DOTALL).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    l, r = s.find("{"), s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]

    s = re.sub(r'(:\s*)\.(\d+)', r'\g<1>0.\2', s)
    s = re.sub(r'(\[|\{)\s*\.(\d+)', r'\g<1>0.\2', s)
    s = re.sub(r',(\s*[}\]])', r'\1', s)

    try:
        return json.loads(s)
    except Exception:
        return {}

# =========================
# Model-only classifier
# =========================
def classify_titles_chatgpt(
    titles: List[str],
    api_key: str,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    neutral_floor: float = 0.55,  # if top score < this → Neutral
    margin_floor: float = 0.10    # if (top-second) < this → Neutral
) -> List[str]:
    """
    Returns list of labels in {'Bullish','Bearish','Neutral'} using model-only reasoning.
    Light guardrails: if low confidence or too close → Neutral.
    """
    if not api_key:
        raise RuntimeError("Missing OpenAI API key in secrets.")

    oai = OpenAI(api_key=api_key)

    SYS = (
        "You are a STRICT sentiment classifier for crypto/finance video titles.\n"
        "Assign exactly one label per title: Bullish, Bearish, or Neutral.\n\n"
        "DEFINITIONS:\n"
        "- Bullish: clear upside language or expectation (rally, surge, breakout, rebound, squeeze, etc.).\n"
        "- Bearish: clear downside language or expectation (drop, dump, crash, plunge, liquidation, breakdown, etc.).\n"
        "- Neutral: tutorials/education, ads/sponsors, generic news, or mixed/hedged titles such as "
        "'X soars but Y crashes' or 'Is this the top?'.\n\n"
        "RULES:\n"
        "1) If signals conflict, choose Neutral.\n"
        "2) Question marks or hedging words (might/could/maybe) → Neutral unless obviously one-sided.\n"
        "3) Clickbait adjectives/emojis do not determine direction.\n\n"
        "OUTPUT JSON ONLY:\n"
        "{\"items\":[{\"index\":0, \"label\":\"Bullish|Bearish|Neutral\", "
        "\"scores\":{\"Bullish\":0.0,\"Bearish\":0.0,\"Neutral\":0.0}}]}\n"
    )

    FEW_SHOT = [
        {
            "role": "user",
            "content": json.dumps([
                {"index": 0, "title": "Bitcoin soars BUT altcoins crash hard"},
                {"index": 1, "title": "Is this the final Bitcoin top?"},
                {"index": 2, "title": "Ethereum down 12% after liquidation cascade"}
            ], ensure_ascii=False)
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "items": [
                    {"index": 0, "label": "Neutral",
                     "scores": {"Bullish": 0.35, "Bearish": 0.35, "Neutral": 0.30}},
                    {"index": 1, "label": "Neutral",
                     "scores": {"Bullish": 0.30, "Bearish": 0.25, "Neutral": 0.45}},
                    {"index": 2, "label": "Bearish",
                     "scores": {"Bullish": 0.05, "Bearish": 0.85, "Neutral": 0.10}}
                ]
            })
        }
    ]

    labels: List[str] = []
    BATCH = 20
    for i in range(0, len(titles), BATCH):
        chunk = titles[i:i+BATCH]
        payload = [{"index": j+i, "title": t} for j, t in enumerate(chunk)]

        resp = oai.chat.completions.create(
            model=model_name,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYS},
                *FEW_SHOT,
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            max_tokens=900
        )

        raw = resp.choices[0].message.content or "{}"
        data = _safe_json_loads(raw)

        by_index = {it.get("index"): it for it in (data.get("items") or [])}

        for j, _ in enumerate(chunk):
            item = by_index.get(j+i, {})
            scrs = (item.get("scores") or {})
            b = float(scrs.get("Bullish", 0.0))
            d = float(scrs.get("Bearish", 0.0))
            n = float(scrs.get("Neutral", 0.0))

            if (b + d + n) <= 0:
                lbl = item.get("label")
                labels.append(lbl if lbl in {"Bullish","Bearish","Neutral"} else "Neutral")
                continue

            scores = {"Bullish": b, "Bearish": d, "Neutral": n}
            top_label = max(scores, key=scores.get)
            top = scores[top_label]
            second = max(v for k, v in scores.items() if k != top_label)

            if top < neutral_floor or (top - second) < margin_floor:
                labels.append("Neutral")
            else:
                labels.append(top_label)

    if len(labels) != len(titles):
        labels += ["Neutral"] * (len(titles) - len(labels))

    return labels

# =========================
# Index computation
# =========================
def compute_index(df: pd.DataFrame, labels: List[str]) -> Tuple[float, pd.DataFrame]:
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    df["Sentiment"] = labels
    df["SentimentScore"] = df["Sentiment"].map(score_map)
    idx_val = float(df["SentimentScore"].mean()) if len(df) else 0.0
    counts = pd.Series(labels).value_counts().reindex(["Bullish","Bearish","Neutral"]).fillna(0).astype(int)
    counts_tbl = pd.DataFrame({"Category": counts.index, "Count (1-last)": counts.values})
    return idx_val, counts_tbl

def zone_color(score: float) -> Tuple[str, str]:
    """Returns (zone_text, CSS color) for the big overall number."""
    if score > 0.4:
        return "Greed", "#2ca02c"  # green
    if score < 0.0:
        return "Fear", "#d62728"   # red
    return "Neutral", "#222222"    # black/gray

# =========================
# Sidebar (range inputs)
# =========================
with st.sidebar:
    st.header("Inputs")

    channel = st.text_input("Channel handle", value="@DataDash")

    now = dt.datetime.utcnow()

    # Start
    st.subheader("Start")
    start_month_name = st.selectbox("Start month", list(calendar.month_name)[1:], index=now.month-1, key="sm")
    start_year = st.number_input("Start year", min_value=2015, max_value=2035, value=max(2019, now.year-1), step=1)

    # End
    st.subheader("End")
    end_month_name = st.selectbox("End month", list(calendar.month_name)[1:], index=now.month-1, key="em")
    end_year = st.number_input("End year", min_value=2015, max_value=2035, value=now.year, step=1)

    min_minutes = st.number_input("Min video minutes", min_value=1, max_value=120, value=5, step=1,
                                  help="Videos shorter than this are excluded.")

    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o"], index=0)

    run = st.button("Run")

start_month = list(calendar.month_name)[1:].index(start_month_name) + 1
end_month   = list(calendar.month_name)[1:].index(end_month_name) + 1

# =========================
# Main Run
# =========================
if run:
    # Validate range
    if (end_year, end_month) < (start_year, start_month):
        st.error("End month must be the same as or after the start month.")
        st.stop()

    # Fetch
    try:
        df = fetch_channel_videos_for_range(
            channel, YOUTUBE_KEY,
            int(start_year), int(start_month),
            int(end_year),   int(end_month),
            int(min_minutes) * 60
        )
    except Exception as e:
        st.error(f"YouTube fetch failed: {e}")
        st.stop()

    if df.empty:
        st.warning("No (>= min minutes) videos found across the selected range.")
        st.stop()

    st.success(f"Found {len(df)} videos across the selected range.")
    st.write("Classifying with ChatGPT…")

    # Classify all titles
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

    # Overall index
    idx_val, counts_tbl = compute_index(df, labels)

    # --- Big overall title (color-coded) ---
    zone, color = zone_color(idx_val)
    st.markdown(
        f"<h2 style='margin-top:0.5rem;'>Overall Sentiment Index</h2>"
        f"<div style='font-size:40px; font-weight:700; color:{color}; margin:-10px 0 12px 2px;'>"
        f"{idx_val:+.2f} • {zone}"
        f"</div>",
        unsafe_allow_html=True
    )

    # Topline
    topline = pd.DataFrame(
        [
            ["Total Videos", int(len(df))],
            ["Total Views", int(df["Views"].sum()) if len(df) else 0],
            ["Avg Views/Video", round(float(df["Views"].mean()), 2) if len(df) else 0.0],
            ["Sentiment Index (−1..+1)", round(idx_val, 3)],
        ],
        columns=["Metric", "Value"],
    )

    # Monthly breakdown (counts & line)
    df_month = df.copy()
    df_month["Sentiment"] = labels
    month_idx = (df_month.groupby("YearMonth")["Sentiment"]
                 .apply(lambda s: pd.Series({"Index": (s=="Bullish").mean() - (s=="Bearish").mean()}))
                 .reset_index())

    # Layout
    c1, c2 = st.columns([1,1])
    with c1:
        st.subheader("Topline")
        st.dataframe(topline, use_container_width=True, hide_index=True)

        st.subheader("Category counts (1-last) — Overall")
        st.dataframe(counts_tbl, use_container_width=True, hide_index=True)

    with c2:
        # Monthly line chart
        st.subheader("Monthly Sentiment Index (line)")
        if not month_idx.empty:
            chart = alt.Chart(month_idx).mark_line(point=True).encode(
                x=alt.X("YearMonth:N", sort=None, title="Month"),
                y=alt.Y("Index:Q", title="Index (−1..+1)", scale=alt.Scale(domain=[-1, 1]))
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No monthly data to plot.")

    # Monthly counts table
    def count_triplet(s):
        return pd.Series({
            "Bullish": int((s=="Bullish").sum()),
            "Bearish": int((s=="Bearish").sum()),
            "Neutral": int((s=="Neutral").sum())
        })
    monthly_counts = (df_month.groupby("YearMonth")["Sentiment"]
                      .apply(count_triplet).reset_index())
    st.subheader("Monthly breakdown (counts)")
    st.dataframe(monthly_counts, use_container_width=True, hide_index=True)

    # Labeled videos table
    st.subheader("Labeled Videos")
    prev = df.drop(columns=["VideoId"], errors="ignore").copy()
    prev.insert(0, "No.", range(1, len(prev)+1))
    prev["Sentiment"] = labels
    st.dataframe(prev, use_container_width=True, height=520, hide_index=True)

    # Downloads (unique keys)
    csv_buf = io.StringIO()
    prev.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV",
        data=csv_buf.getvalue(),
        file_name=f"{channel.strip('@').lower()}_{start_year}-{start_month:02d}_to_{end_year}-{end_month:02d}.csv",
        mime="text/csv",
        key="dl_csv_range",
    )

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as xw:
        prev.to_excel(xw, index=False, sheet_name="Videos")
        topline.to_excel(xw, index=False, sheet_name="Topline")
        counts_tbl.to_excel(xw, index=False, sheet_name="SentimentCounts")
        month_idx.to_excel(xw, index=False, sheet_name="MonthlyIndex")
        monthly_counts.to_excel(xw, index=False, sheet_name="MonthlyCounts")
    st.download_button(
        "Download Excel",
        data=xlsx_buf.getvalue(),
        file_name=f"{channel.strip('@').lower()}_{start_year}-{start_month:02d}_to_{end_year}-{end_month:02d}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_xlsx_range",
    )
