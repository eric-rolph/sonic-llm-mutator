import difflib
import json
import os

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.dashboard_data import (
    beaten_acts,
    champion_entry,
    chart_caption,
    chart_series,
    diagnosis_freshness,
    frontier_summary,
    is_new_champion,
    learned_moves,
    run_liveness,
    stagnation_status,
    video_caption,
)
from core.evaluator import FITNESS_FORMULA

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Sonic LLM Mutator Dashboard", page_icon="🦔")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="datarefresh")

st.title("🦔 Sonic AI: LLM Evolutionary Genetic Algorithm")

# Resolve everything relative to this file so the dashboard works no matter
# which directory streamlit was launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HISTORY_PATH = os.path.join(BASE_DIR, "artifacts", "history.json")
CHAMPION_PATH = os.path.join(BASE_DIR, "policies", "champion_policy.py")
CHAMPION_VIDEO_PATH = os.path.join(BASE_DIR, "artifacts", "videos", "champion.mp4")
LATEST_VIDEO_PATH = os.path.join(BASE_DIR, "artifacts", "videos", "latest.mp4")
DIAGNOSIS_REPORT_PATH = os.path.join(BASE_DIR, "artifacts", "diagnosis", "latest_report.json")


def resolve_artifact(path):
    """History/diagnosis files record repo-relative paths; resolve them."""
    if not path:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_data():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
    return []


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def load_video_bytes(path, mtime, size):
    """Cache video bytes by (path, mtime, size).

    The autorefresh reruns this script every few seconds; re-reading and
    re-hashing a 10+ MB mp4 each run made the page take longer to render
    than the refresh interval. max_entries bounds the cache: without it, every
    replaced video left its ~12 MB predecessor in memory for the whole
    multi-hour run (agency review).
    """
    with open(path, "rb") as f:
        return f.read()


def show_video(path, waiting_message, caption=None):
    if not os.path.exists(path):
        st.info(waiting_message)
        return
    try:
        stat = os.stat(path)
        st.video(load_video_bytes(path, stat.st_mtime, stat.st_size), format="video/mp4")
        if caption:
            st.caption(caption)
    except PermissionError:
        st.warning("Video is currently rendering (file locked by ffmpeg). It will appear on the next refresh!")
    except OSError as e:
        st.error(f"Error reading video: {e}")


history_data = load_data()
latest = history_data[-1] if history_data else None
best = champion_entry(history_data)
comps = (latest or {}).get("components", {})
frontier = frontier_summary(comps)

# --- Liveness header: is the run even alive? -------------------------------
liveness = run_liveness(latest)
if latest is None:
    st.info(
        "No run data yet. Start the evolutionary pipeline with `python main.py` "
        "and this dashboard will come alive."
    )
else:
    st.caption(f"Generation {latest.get('generation', '?')} · {liveness['text']}")
    if liveness["stale"]:
        st.warning(
            f"No new generation for {liveness['minutes']} minutes — the training run "
            "may have stopped. This page only reflects the last recorded state."
        )

# --- New-champion celebration (once per record, not per refresh) ------------
if latest is not None and is_new_champion(history_data):
    st.success(
        f"🏆 NEW ALL-TIME CHAMPION — generation {latest.get('generation', '?')} "
        f"at {latest.get('fitness', 0):,.2f}!"
    )
    if st.session_state.get("celebrated_generation") != latest.get("generation"):
        st.session_state["celebrated_generation"] = latest.get("generation")
        st.balloons()

# --- Glance row: the run state in one line of tiles -------------------------
if latest is not None:
    tiles = st.columns(6)
    tiles[0].metric("Champion 🏆", f"{(best or {}).get('fitness', 0):,.0f}")
    tiles[1].metric(
        "Latest Attempt 🧪",
        f"{latest.get('fitness', 0):,.0f}",
        delta=f"{latest.get('fitness', 0) - (best or {}).get('fitness', 0):,.0f} vs champion",
    )
    if frontier:
        tiles[2].metric("Frontier 📍", f"x={frontier['x']:,}", delta=frontier["label"], delta_color="off")
    else:
        tiles[2].metric("Frontier 📍", "—")
    tiles[3].metric("Acts Beaten 🏁", int(comps.get("levels_cleared", 0)))
    tiles[4].metric("Speed Bonus", f"{comps.get('speed', 0):,.0f}")
    tiles[5].metric("Rings/Pts", f"{comps.get('rings', 0) + comps.get('score', 0):,.0f}")

    # --- The boss fight: where the champion currently dies ------------------
    if frontier:
        trophies = beaten_acts(comps)
        fight = f"**Now fighting:** {frontier['label']} — dies at x={frontier['x']:,}"
        if trophies:
            fight += "  ·  **Beaten:** " + ", ".join(trophies)
        st.markdown(fight)
        if frontier.get("completion_target"):
            st.progress(frontier["progress"])
            st.caption(
                f"{frontier['x']:,} / {frontier['completion_target']:,} to the act sign "
                f"({frontier['progress']:.0%})"
            )

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📈 Fitness Progression")
    if history_data:
        series = chart_series(history_data)
        df = pd.DataFrame(
            {"attempt": series["attempt"], "champion": series["champion"]},
            index=series["generation"],
        )
        st.line_chart(df)
        st.caption(chart_caption(history_data))

        vc1, vc2 = st.columns(2)
        with vc1:
            st.subheader("🏆 Champion Run")
            show_video(
                CHAMPION_VIDEO_PATH,
                "No champion video rendered yet. Waiting for a successful run...",
                caption=video_caption("Champion", best),
            )
        with vc2:
            st.subheader("🧪 Latest Attempt")
            show_video(
                LATEST_VIDEO_PATH,
                "No latest video rendered yet.",
                caption=video_caption("Latest attempt", latest),
            )

        # Routine per-generation status is INFORMATION, not an error: the old
        # always-red box trained users to ignore red (agency review).
        st.markdown("**How the latest attempt ended:**")
        st.info(latest.get("failure_reason") or "No failure recorded.")

        with st.expander("🤖 LLM reasoning for the latest mutation"):
            st.write(latest.get("llm_reasoning", "No reasoning provided."))

        diagnosis = load_json(DIAGNOSIS_REPORT_PATH)
        if diagnosis and diagnosis.get("report"):
            st.markdown("**🔬 Frontier intel (sweep + agentic diagnosis):**")
            freshness = diagnosis_freshness(diagnosis)
            if freshness:
                st.caption(
                    f"Report {freshness}, for: {diagnosis.get('failure_reason', 'unknown failure')}"
                )
            st.info(diagnosis["report"])
            experiments = diagnosis.get("verified_experiments") or []
            if experiments:
                st.markdown("**Verified escapes (measured in the emulator):**")
                st.table(
                    [
                        {
                            "input": e.get("actions", "?"),
                            "from x": e.get("start_x", "?"),
                            "reached x": e.get("max_x", "?"),
                            "frames": e.get("hold_frames", "?"),
                        }
                        for e in experiments[:5]
                    ]
                )
            evidence = resolve_artifact(diagnosis.get("evidence_screenshot"))
            if evidence and os.path.exists(evidence):
                st.image(evidence, caption="Diagnosis evidence frame (from interactive replay)")
    else:
        st.info("No history data available yet.")

with col2:
    st.subheader("💻 AI Generated Python Policies")

    # Load code strings
    champ_code = ""
    if os.path.exists(CHAMPION_PATH):
        with open(CHAMPION_PATH, "r") as f:
            champ_code = f.read()

    latest_code = ""
    archive_path = ""
    if history_data:
        archive_path = resolve_artifact(history_data[-1].get("archive_path", "")) or ""
        if os.path.exists(archive_path):
            with open(archive_path, "r") as f:
                latest_code = f.read()

    moves = learned_moves(champ_code)
    if moves:
        with st.expander(f"🕹️ Moves Sonic learned ({len(moves)})", expanded=False):
            for move in moves:
                st.markdown(f"- {move['label']}")

    tab1, tab2, tab3 = st.tabs(["🏆 Champion Policy", "🧪 Latest Mutation", "🔍 Active Diff"])

    with tab1:
        if champ_code:
            st.code(champ_code, language="python")
        else:
            st.info("Champion policy file not found.")

    with tab2:
        if latest_code:
            st.code(latest_code, language="python")
        else:
            st.info(f"Latest policy file not found at {archive_path}")

    with tab3:
        if champ_code and latest_code:
            champ_lines = champ_code.splitlines(keepends=True)
            latest_lines = latest_code.splitlines(keepends=True)
            diff = list(difflib.unified_diff(champ_lines, latest_lines, fromfile='Champion', tofile='Latest', n=3))
            if diff:
                st.code("".join(diff), language="diff")
            else:
                st.info("No differences found between Champion and Latest policy. (They are identical)")
        else:
            st.info("Need both Champion and Latest files to compute diff.")

# Sidebar renders unconditionally: on a fresh clone the old version showed
# nothing at all, giving no hint of how the system scores runs (agency review).
with st.sidebar:
    st.header("🧮 Fitness Calculation")
    st.markdown(f"`{FITNESS_FORMULA}`")
    st.caption(
        "The AI is highly incentivized to move right as fast as possible, with small "
        "bonuses for rings/points. A one-off completion bonus is awarded for reaching "
        "the level's end zone; each act fully cleared is worth far more."
    )

    st.header("⚠️ Stagnation Monitor")
    if latest is not None:
        status = stagnation_status(latest.get("stagnation_counter", 0))
        getattr(st, status["level"])(status["text"])
    else:
        st.info("Waiting for the first generation.")

st.markdown("---")
st.caption("Powered by LM Studio, gym-retro & a lot of verified emulator experiments")
