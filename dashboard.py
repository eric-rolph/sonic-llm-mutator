import streamlit as st
import json
import os
import time
import pandas as pd
from streamlit_autorefresh import st_autorefresh

# Set page layout to wide
st.set_page_config(layout="wide", page_title="Sonic LLM Mutator Dashboard", page_icon="🦔")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="datarefresh")

st.title("🦔 Sonic AI: LLM Evolutionary Genetic Algorithm")
st.markdown("Watching an LLM dynamically write and mutate Python code to beat Sonic the Hedgehog.")

HISTORY_PATH = "artifacts/history.json"
CHAMPION_PATH = "policies/champion_policy.py"
CHAMPION_VIDEO_PATH = "artifacts/videos/champion.mp4"
LATEST_VIDEO_PATH = "artifacts/videos/latest.mp4"

# Load History Data
def load_data():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                data = json.load(f)
                return data
        except:
            return []
    return []

history_data = load_data()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📺 The Champion Run")
    if os.path.exists(CHAMPION_VIDEO_PATH):
        try:
            with open(CHAMPION_VIDEO_PATH, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")
        except PermissionError:
            st.warning("Video is currently rendering (file locked by ffmpeg). It will appear on the next 5-second refresh!")
        except Exception as e:
            st.error(f"Error reading champion video: {e}")
    else:
        st.info("No champion video rendered yet. Waiting for a successful run...")

    st.subheader("📺 Latest Generation Attempt")
    if os.path.exists(LATEST_VIDEO_PATH):
        try:
            with open(LATEST_VIDEO_PATH, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")
        except PermissionError:
            st.warning("Video is currently rendering (file locked by ffmpeg). It will appear on the next 5-second refresh!")
        except Exception as e:
            st.error(f"Error reading latest video: {e}")
    else:
        st.info("No latest video rendered yet.")

    st.subheader("📈 Fitness Progression")
    if history_data:
        df = pd.DataFrame(history_data)
        # Plot Fitness over generations
        st.line_chart(df.set_index("generation")["fitness"])
        
        latest = history_data[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Champion Fitness", f"{latest['fitness']:.2f}")
        
        comps = latest.get("components", {})
        c2.metric("Distance Score", f"{comps.get('distance', 0):.2f}")
        c3.metric("Speed Bonus", f"{comps.get('speed', 0):.2f}")
        c4.metric("Rings/Points", f"{comps.get('rings', 0) + comps.get('score', 0):.2f}")
        
        st.markdown("**LLM Reasoning for latest mutation:**")
        st.info(latest.get("llm_reasoning", "No reasoning provided."))
        
        st.markdown("**Failure Reason:**")
        st.error(latest.get("failure_reason", "Unknown"))
        
        with st.sidebar:
            st.header("🧮 Fitness Calculation")
            st.markdown("`fitness = (distance * 2) + ((distance / frames) * 100) + (rings * 10) + score`")
            st.caption("The AI is highly incentivized to move right as fast as possible, with small bonuses for rings/points.")
            
            st.header("⚠️ Stagnation Monitor")
            stag_count = latest.get("stagnation_counter", 0)
            if stag_count > 3:
                st.error(f"Stagnation Level: {stag_count}/5\nA blankRestart mutation is imminent!")
            elif stag_count > 0:
                st.warning(f"Stagnation Level: {stag_count}/5\nThe AI is struggling to beat the champion.")
            else:
                st.success(f"Stagnation Level: 0/5\nMaking steady progress!")
    else:
        st.info("No history data available yet.")

with col2:
    st.subheader("💻 AI Generated Python Policies")
    
    tab1, tab2 = st.tabs(["🏆 Champion Policy", "🧪 Latest Mutation"])
    
    with tab1:
        if os.path.exists(CHAMPION_PATH):
            with open(CHAMPION_PATH, "r") as f:
                code = f.read()
            st.code(code, language="python")
        else:
            st.info("Champion policy file not found.")
            
    with tab2:
        # Find the latest archive file based on the history
        if history_data:
            latest = history_data[-1]
            archive_path = latest.get("archive_path", "")
            if os.path.exists(archive_path):
                with open(archive_path, "r") as f:
                    latest_code = f.read()
                st.code(latest_code, language="python")
            else:
                st.info(f"Latest policy file not found at {archive_path}")
        else:
            st.info("No history yet.")

st.markdown("---")
st.caption("Powered by Qwen3.6-27B (LM Studio) & stable-retro")
