import os
from datetime import datetime

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="TitanicBot",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

:root {
  --bg:      #0d1117;
  --card:    #161b22;
  --border:  #30363d;
  --primary: #58a6ff;
  --accent:  #f78166;
  --text:    #e6edf3;
  --muted:   #8b949e;
  --purple:  #bc8cff;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 860px; }

.hero { text-align: center; padding: 2.4rem 1rem 1.6rem; }
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 60%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero p { color: var(--muted); font-size: 0.95rem; margin-top: 0.5rem; }

.stat-row { display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap; margin-bottom: 1.6rem; }
.stat-pill {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-size: 0.8rem;
    color: var(--muted);
    display: flex; align-items: center; gap: 0.4rem;
}
.stat-pill span.val { color: var(--primary); font-weight: 600; }

.msg-wrap { display: flex; gap: 0.75rem; margin-bottom: 1rem; align-items: flex-start; }
.msg-wrap.user { flex-direction: row-reverse; }
.avatar {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; border: 1.5px solid var(--border);
}
.avatar.bot  { background: linear-gradient(135deg,#58a6ff33,#bc8cff33); border-color: #58a6ff55; }
.avatar.user { background: linear-gradient(135deg,#f7816633,#ffa65733); border-color: #f7816655; }
.bubble {
    max-width: 82%;
    padding: 0.75rem 1rem;
    border-radius: 14px;
    font-size: 0.92rem;
    line-height: 1.6;
    border: 1px solid var(--border);
}
.bubble.bot  { background: var(--card); border-color: #30363d; border-top-left-radius: 4px; }
.bubble.user { background: #1c2d3f; border-color: #1f6feb55; border-top-right-radius: 4px; }
.bubble p { margin: 0; }
.ts { font-size: 0.68rem; color: var(--muted); margin-top: 0.35rem; }

.chart-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.6rem 0 0.2rem;
}
.chart-caption { font-size: 0.78rem; color: var(--muted); text-align: center; margin-top: 0.4rem; }

.stChatInput > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stChatInput textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

.typing { display: flex; gap: 5px; padding: 0.5rem 0; align-items: center; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--primary); animation: bounce 1.2s infinite; }
.dot:nth-child(2) { animation-delay: .2s; background: var(--purple); }
.dot:nth-child(3) { animation-delay: .4s; background: var(--accent); }
@keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-8px)} }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL)

    st.markdown("---")
    st.markdown("### 📊 About the Dataset")
    st.markdown(
        "The **Titanic dataset** contains info on 891 passengers — "
        "survival, class, sex, age, fare, and embarkation port."
    )

    st.markdown("---")
    st.markdown("### 💡 Try asking…")
    suggestions = [
        "What was the survival rate?",
        "Show me a histogram of passenger ages",
        "Average fare by passenger class",
        "How many passengers from each port?",
        "Survival rate by gender",
        "Show age distribution by class as a box plot",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=f"sb_{suggestion}", use_container_width=True):
            st.session_state.pending = suggestion

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div class="hero">
  <h1>🚢 TitanicBot</h1>
  <p>Ask anything about the Titanic passenger dataset — get answers and charts instantly.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
  <div class="stat-pill">👥 Passengers <span class="val">891</span></div>
  <div class="stat-pill">✅ Survived <span class="val">342</span></div>
  <div class="stat-pill">💰 Avg Fare <span class="val">$32.20</span></div>
  <div class="stat-pill">🌍 Ports <span class="val">S · C · Q</span></div>
</div>
""", unsafe_allow_html=True)


def show_message(role, content, image_b64=None, image_caption=None, ts=None):
    side = "user" if role == "user" else "bot"
    icon = "👤" if role == "user" else "🚢"

    st.markdown(f"""
    <div class="msg-wrap {side}">
      <div class="avatar {side}">{icon}</div>
      <div>
        <div class="bubble {side}"><p>{content}</p></div>
        <div class="ts">{ts or ""}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if image_b64:
        st.markdown(
            f'<div class="chart-card">'
            f'<img src="data:image/png;base64,{image_b64}" style="width:100%;border-radius:8px;"/>'
            f'<div class="chart-caption">{image_caption or ""}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


for msg in st.session_state.messages:
    show_message(
        role=msg["role"],
        content=msg["content"],
        image_b64=msg.get("image_b64"),
        image_caption=msg.get("image_caption"),
        ts=msg.get("ts"),
    )

user_input = st.chat_input("Ask about the Titanic dataset…")

if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None

if user_input:
    now = datetime.now().strftime("%H:%M")

    st.session_state.messages.append({"role": "user", "content": user_input, "ts": now})
    show_message("user", user_input, ts=now)

    typing = st.empty()
    typing.markdown(
        '<div class="bubble bot" style="display:inline-block;margin-bottom:.5rem;">'
        '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        res = requests.post(f"{backend_url}/chat", json={"question": user_input}, timeout=90)
        res.raise_for_status()
        data = res.json()
        answer = data.get("answer", "Something went wrong, please try again.")
        image_b64 = data.get("image_b64")
        image_caption = data.get("image_caption")
    except requests.exceptions.ConnectionError:
        answer = f"⚠️ Can't reach the backend at `{backend_url}`. Make sure it's running."
        image_b64 = None
        image_caption = None
    except Exception as e:
        answer = f"⚠️ {e}"
        image_b64 = None
        image_caption = None

    typing.empty()

    reply_ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "image_b64": image_b64,
        "image_caption": image_caption,
        "ts": reply_ts,
    })
    show_message("assistant", answer, image_b64, image_caption, ts=reply_ts)
    st.rerun()
