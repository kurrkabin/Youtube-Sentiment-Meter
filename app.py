import json
import uuid
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
import streamlit as st

# --------------------------- Config ---------------------------
st.set_page_config(page_title="Sport Trading Reminders (UTC)", page_icon="⏰", layout="wide")

SPORTS = [
    "Cricket",
    "Darts",
    "Rugby Union",
    "Rugby League",
    "MotorSports",
    "Aussie Rules",
    "Boxing",
    "Snooker",
]

DATA_PATH = Path("tasks.json")

# Invisible auto-check cadence (milliseconds). Default: 5 minutes.
AUTO_REFRESH_MS = 300_000

# --------------------------- Helpers ---------------------------
def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")

def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)

def load_tasks() -> list[dict]:
    if DATA_PATH.exists():
        try:
            return json.loads(DATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_tasks(tasks: list[dict]) -> None:
    DATA_PATH.write_text(json.dumps(tasks, indent=2), encoding="utf-8")

def ensure_state():
    if "tasks" not in st.session_state:
        st.session_state.tasks = load_tasks()

def add_task(sport: str, txt: str, dt_utc: datetime):
    t = {
        "id": str(uuid.uuid4()),
        "sport": sport,
        "text": txt.strip(),
        "when_utc": iso(dt_utc),
        "created_utc": iso(now_utc()),
        "done": False,
        "alerted": False,
        "snoozed_minutes": 0,
    }
    st.session_state.tasks.append(t)
    save_tasks(st.session_state.tasks)

def snooze_task(task_id: str, minutes: int = 5):
    for t in st.session_state.tasks:
        if t["id"] == task_id:
            when = parse_iso(t["when_utc"])
            t["when_utc"] = iso(when + timedelta(minutes=minutes))
            t["alerted"] = False
            t["snoozed_minutes"] = t.get("snoozed_minutes", 0) + minutes
            break
    save_tasks(st.session_state.tasks)

def mark_done(task_id: str):
    for t in st.session_state.tasks:
        if t["id"] == task_id:
            t["done"] = True
            break
    save_tasks(st.session_state.tasks)

def delete_task(task_id: str):
    st.session_state.tasks = [t for t in st.session_state.tasks if t["id"] != task_id]
    save_tasks(st.session_state.tasks)

def format_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

def due_status(t: dict, nowt: datetime) -> str:
    if t["done"]:
        return "✅ Done"
    when = parse_iso(t["when_utc"])
    if nowt >= when:
        return "🔔 Due"
    else:
        mins = int((when - nowt).total_seconds() // 60)
        return f"⏳ In {mins} min"

def live_utc_clock():
    # Renders a client-side live UTC clock (no reruns needed)
    st.markdown(
        """
        <div style="display:flex;justify-content:flex-end;">
          <div id="utc-clock" style="text-align:right;font-variant-numeric:tabular-nums;">
              <span style="opacity:.7;margin-right:.5rem;">UTC</span>
              <strong id="utc-time">--:--:--</strong>
          </div>
        </div>
        <script>
          function pad(n){return n.toString().padStart(2,'0');}
          function tick(){
            const d = new Date();
            const y = d.getUTCFullYear();
            const m = pad(d.getUTCMonth()+1);
            const day = pad(d.getUTCDate());
            const hh = pad(d.getUTCHours());
            const mm = pad(d.getUTCMinutes());
            const ss = pad(d.getUTCSeconds());
            const s = `${y}-${m}-${day} ${hh}:${mm}:${ss}`;
            const el = document.getElementById("utc-time");
            if (el) el.textContent = s;
          }
          tick();
          setInterval(tick, 1000);
        </script>
        """,
        unsafe_allow_html=True,
    )

def play_beep_web_audio():
    """Use Web Audio API to generate a short beep—reliable across browsers once tab is interacted with."""
    st.components.v1.html(
        """
        <script>
          (function() {
            try {
              const ctx = new (window.AudioContext || window.webkitAudioContext)();
              const osc = ctx.createOscillator();
              const gain = ctx.createGain();
              osc.type = 'sine';
              osc.frequency.value = 880; // A5
              osc.connect(gain);
              gain.connect(ctx.destination);
              gain.gain.setValueAtTime(0.0001, ctx.currentTime);
              gain.gain.exponentialRampToValueAtTime(0.3, ctx.currentTime + 0.01);
              osc.start();
              gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
              osc.stop(ctx.currentTime + 0.27);
              if (ctx.state === 'suspended') { ctx.resume(); }
            } catch(e) { console.log('Beep error:', e); }
          })();
        </script>
        """,
        height=0,
    )

# --------------------------- App ---------------------------
ensure_state()

st.title("⏰ Sport Trading Reminders (UTC)")

# Top-right: live UTC clock. No refresh UI.
live_utc_clock()

# Invisible server-side auto-check every 5 minutes (no visible widget)
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=AUTO_REFRESH_MS, key="auto_refresh_5m", limit=None)
except Exception:
    pass  # If the helper isn't installed, the app still works; you'll interact occasionally.

st.markdown("---")

# Adders per sport
st.subheader("Add reminders (UTC)")
today_utc = now_utc().date()
for sport in SPORTS:
    with st.expander(f"➕ {sport}", expanded=False):
        c1, c2, c3, c4 = st.columns([1.0, 1.0, 2.2, 1.2])

        with c1:
            d = st.date_input(f"Date (UTC) – {sport}", value=today_utc, key=f"{sport}_date")
        with c2:
            tm = st.time_input(
                f"Time (UTC) – {sport}",
                value=time(0, 0),
                key=f"{sport}_time",
                step=timedelta(minutes=5),  # 5-minute increments
            )

        with c3:
            txt = st.text_input(
                f"Action / note – {sport}",
                placeholder="e.g., goes live; freeze groups; freeze main market; settle score; trade live…",
                key=f"{sport}_text",
            )

        with c4:
            if st.button("Add", key=f"{sport}_add"):
                if txt.strip():
                    dt_utc = datetime.combine(d, tm).replace(tzinfo=timezone.utc)
                    add_task(sport, txt, dt_utc)
                    st.success(f"Added for {sport} at {format_dt(dt_utc)}")
                else:
                    st.warning("Please enter an action/note.")

st.markdown("---")

# Boards
nowt = now_utc()
tasks_sorted = sorted(st.session_state.tasks, key=lambda t: (parse_iso(t["when_utc"]), t["sport"]))

due_tasks = [t for t in tasks_sorted if (not t["done"]) and nowt >= parse_iso(t["when_utc"])]
upcoming_tasks = [t for t in tasks_sorted if (not t["done"]) and nowt < parse_iso(t["when_utc"])]
done_tasks = [t for t in tasks_sorted if t["done"]]

# Fire a beep exactly when we see newly-due, unalerted tasks
newly_due = [t for t in due_tasks if not t.get("alerted", False)]
if newly_due:
    for t in newly_due:
        t["alerted"] = True
    save_tasks(st.session_state.tasks)
    play_beep_web_audio()
    st.toast(f"🔔 {len(newly_due)} reminder(s) due now", icon="🔔")

st.subheader("🔔 Due now")
if not due_tasks:
    st.info("No due reminders at the moment.")
else:
    for t in due_tasks:
        with st.container(border=True):
            when = parse_iso(t["when_utc"])
            st.markdown(f"**{t['sport']}** — {t['text']}")
            st.caption(f"Scheduled: {format_dt(when)}")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.button("✅ Mark done", key=f"done_{t['id']}", on_click=mark_done, args=(t["id"],))
            with c2:
                st.button("⏱️ Snooze +5m", key=f"snooze_{t['id']}", on_click=snooze_task, args=(t["id"], 5))
            with c3:
                st.button("🗑️ Delete", key=f"del_{t['id']}", on_click=delete_task, args=(t["id"],))

st.subheader("⏳ Upcoming (UTC)")
if not upcoming_tasks:
    st.info("Nothing upcoming.")
else:
    for t in upcoming_tasks:
        with st.container(border=True):
            when = parse_iso(t["when_utc"])
            st.markdown(f"**{t['sport']}** — {t['text']}")
            st.caption(f"Scheduled: {format_dt(when)} • {due_status(t, nowt)} • Snoozed: {t.get('snoozed_minutes',0)} min")
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                st.button("✅ Mark done", key=f"udone_{t['id']}", on_click=mark_done, args=(t["id"],))
            with c2:
                st.button("⏱️ Snooze +5m", key=f"usnooze_{t['id']}", on_click=snooze_task, args=(t["id"], 5))
            with c3:
                st.button("🗑️ Delete", key=f"udel_{t['id']}", on_click=delete_task, args=(t["id"],))

with st.expander("✔️ Completed"):
    if not done_tasks:
        st.caption("No completed items yet.")
    else:
        for t in done_tasks:
            when = parse_iso(t["when_utc"])
            st.write(f"• **{t['sport']}** — {t['text']}  _(scheduled {format_dt(when)})_")

st.markdown("---")
st.caption("Note: Some browsers require one click on the page before audio can play.")
