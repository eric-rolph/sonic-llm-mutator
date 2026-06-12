import json
import os

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from core.evaluator import FITNESS_FORMULA

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Sonic LLM Mutator Dashboard", page_icon="🦔")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="datarefresh")

st.title("🦔 Sonic AI: LLM Evolutionary Genetic Algorithm")
st.markdown("Watching an LLM dynamically write and mutate Python code to beat Sonic the Hedgehog.")

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


# Load History Data
def load_data():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                data = json.load(f)
                return data
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


@st.cache_data(show_spinner=False)
def load_video_bytes(path, mtime, size):
    """Cache video bytes by (path, mtime, size).

    The autorefresh reruns this script every few seconds; re-reading and
    re-hashing a 10+ MB mp4 each run made the page take longer to render
    than the refresh interval. The cache key changes when the renderer
    replaces the file, so new videos still appear.
    """
    with open(path, "rb") as f:
        return f.read()


def show_video(path, waiting_message):
    if not os.path.exists(path):
        st.info(waiting_message)
        return
    try:
        stat = os.stat(path)
        st.video(load_video_bytes(path, stat.st_mtime, stat.st_size), format="video/mp4")
    except PermissionError:
        st.warning("Video is currently rendering (file locked by ffmpeg). It will appear on the next refresh!")
    except OSError as e:
        st.error(f"Error reading video: {e}")


history_data = load_data()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📺 The Champion Run")
    show_video(CHAMPION_VIDEO_PATH, "No champion video rendered yet. Waiting for a successful run...")

    st.subheader("📺 Latest Generation Attempt")
    show_video(LATEST_VIDEO_PATH, "No latest video rendered yet.")

    st.subheader("📈 Fitness Progression")
    if history_data:
        df = pd.DataFrame(history_data)
        # Plot Fitness over generations
        st.line_chart(df.set_index("generation")["fitness"])

        latest = history_data[-1]
        all_time_champion_fitness = max([entry.get('fitness', -1) for entry in history_data]) if history_data else 0

        comps = latest.get("components", {})
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("All-Time Champion Fitness 🏆", f"{all_time_champion_fitness:.2f}")
        c2.metric("Latest Attempt Fitness 🧪", f"{latest['fitness']:.2f}")
        c3.metric("Levels Cleared 🏁", int(comps.get('levels_cleared', 0)))
        c4.metric("Latest Distance", f"{comps.get('distance', 0):.2f}")
        c5.metric("Latest Speed Bonus", f"{comps.get('speed', 0):.2f}")
        c6.metric("Latest Rings/Pts", f"{comps.get('rings', 0) + comps.get('score', 0):.2f}")

        st.markdown("**LLM Reasoning for latest mutation:**")
        st.info(latest.get("llm_reasoning", "No reasoning provided."))

        st.markdown("**Failure Reason:**")
        st.error(latest.get("failure_reason", "Unknown"))

        diagnosis = load_json(DIAGNOSIS_REPORT_PATH)
        if diagnosis and diagnosis.get("report"):
            st.markdown("**🔬 Agentic Failure Diagnosis (latest frontier):**")
            st.info(diagnosis["report"])
            evidence = resolve_artifact(diagnosis.get("evidence_screenshot"))
            if evidence and os.path.exists(evidence):
                st.image(evidence, caption="Diagnosis evidence frame (from interactive replay)")

        with st.sidebar:
            st.header("🧮 Fitness Calculation")
            st.markdown(f"`{FITNESS_FORMULA}`")
            st.caption("The AI is highly incentivized to move right as fast as possible, with small bonuses for rings/points. A one-off completion bonus is awarded for reaching the level's end zone.")

            st.header("⚠️ Stagnation Monitor")
            stag_count = latest.get("stagnation_counter", 0)
            if stag_count > 3:
                st.warning(
                    f"Stagnation Level: {stag_count}/5\n"
                    "The champion will be preserved while the mutator explores a distinct strategy."
                )
            elif stag_count > 0:
                st.warning(f"Stagnation Level: {stag_count}/5\nThe AI is struggling to beat the champion.")
            else:
                st.success("Stagnation Level: 0/5\nMaking steady progress!")
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
            import difflib
            champ_lines = champ_code.splitlines(keepends=True)
            latest_lines = latest_code.splitlines(keepends=True)
            diff = list(difflib.unified_diff(champ_lines, latest_lines, fromfile='Champion', tofile='Latest', n=3))
            if diff:
                st.code("".join(diff), language="diff")
            else:
                st.info("No differences found between Champion and Latest policy. (They are identical)")
        else:
            st.info("Need both Champion and Latest files to compute diff.")

st.markdown("---")
st.caption("Powered by Gemini API, LM Studio & stable-retro")
