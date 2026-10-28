import json
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_DIR = Path(__file__).resolve().parent / "models"

MAX_WORDS = 5000
MAX_LENGTH = 60

SAMPLES = {
    "🎁 Prize scam": "Congratulations! You have won a $1000 Walmart gift card. Click http://bit.ly/claim-now to claim your prize today!",
    "🏦 Bank alert": "URGENT: Your account has been suspended. Verify your details immediately at secure-bank-login.com to avoid closure.",
    "💬 Friend": "Hey, are we still meeting for lunch tomorrow? Let me know what time works for you.",
    "📦 Delivery": "Your package will arrive between 2-4 PM today. Reply 1 to reschedule.",
}

st.set_page_config(
    page_title="SpamShield AI · SMS Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 2. CUSTOM UI STYLING
# ==========================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #070B14;
        --card: rgba(255, 255, 255, 0.04);
        --border: rgba(255, 255, 255, 0.10);
        --muted: #94A3B8;
        --accent: #38BDF8;
        --accent2: #A78BFA;
        --danger: #F87171;
        --safe: #4ADE80;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(900px 500px at 10% -10%, rgba(56,189,248,0.16), transparent 60%),
            radial-gradient(800px 500px at 100% 0%, rgba(167,139,250,0.14), transparent 60%),
            var(--bg);
        color: #E2E8F0;
    }

    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1150px; }

    section[data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.95);
        border-right: 1px solid var(--border);
    }

    /* Hero */
    .hero {
        padding: 34px 36px;
        border-radius: 22px;
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(167,139,250,0.12));
        margin-bottom: 22px;
    }
    .hero h1 {
        margin: 0 0 6px 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38BDF8, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 0; color: var(--muted); font-size: 1.05rem; }
    .pill {
        display: inline-block;
        margin: 14px 8px 0 0;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #CBD5E1;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.05);
    }

    /* Cards */
    .card {
        padding: 20px 22px;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: var(--card);
        height: 100%;
    }
    .card-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        font-weight: 600;
        margin-bottom: 8px;
    }
    .big-number { font-size: 2.1rem; font-weight: 800; line-height: 1.1; }

    /* Verdict banner */
    .verdict {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 22px 26px;
        border-radius: 18px;
        margin: 6px 0 18px 0;
    }
    .verdict .icon { font-size: 2.6rem; }
    .verdict .label { font-size: 1.6rem; font-weight: 800; }
    .verdict .sub { color: var(--muted); font-size: 0.92rem; }
    .verdict.spam {
        background: linear-gradient(135deg, rgba(239,68,68,0.20), rgba(239,68,68,0.06));
        border: 1px solid rgba(239,68,68,0.55);
    }
    .verdict.spam .label { color: var(--danger); }
    .verdict.ham {
        background: linear-gradient(135deg, rgba(34,197,94,0.20), rgba(34,197,94,0.06));
        border: 1px solid rgba(34,197,94,0.55);
    }
    .verdict.ham .label { color: var(--safe); }

    /* Probability meter */
    .meter {
        position: relative;
        height: 14px;
        border-radius: 999px;
        background: linear-gradient(90deg, #22C55E 0%, #FACC15 50%, #EF4444 100%);
        margin: 14px 0 6px 0;
    }
    .meter .marker {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 24px;
        border-radius: 3px;
        background: #fff;
        box-shadow: 0 0 10px rgba(255,255,255,0.8);
    }
    .meter .thr {
        position: absolute;
        top: -2px;
        width: 2px;
        height: 18px;
        background: rgba(0,0,0,0.65);
    }
    .meter-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: var(--muted);
    }

    /* Model cards */
    .tag {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .tag.spam { background: rgba(239,68,68,0.18); color: var(--danger); }
    .tag.ham { background: rgba(34,197,94,0.18); color: var(--safe); }

    /* Widgets */
    .stTextArea textarea {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        color: #F1F5F9 !important;
        font-size: 1rem;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(56,189,248,0.2) !important;
    }
    .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        font-weight: 600;
        transition: all .15s ease;
    }
    .stButton > button:hover {
        border-color: var(--accent);
        transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0EA5E9, #8B5CF6);
        border: none;
        color: white;
        padding: 0.65rem 1rem;
    }
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px 18px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
    h2, h3 { color: #F1F5F9 !important; font-weight: 700 !important; }

    .step {
        display: flex;
        gap: 14px;
        padding: 12px 0;
        border-bottom: 1px solid var(--border);
    }
    .step .num {
        flex: 0 0 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        background: linear-gradient(135deg, #0EA5E9, #8B5CF6);
        color: white;
    }
    .step .txt b { color: #F1F5F9; }
    .step .txt span { color: var(--muted); font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD MODELS AND VOCABULARY
# ==========================================

@st.cache_resource
def load_resources():

    ann = tf.keras.models.load_model(
        MODEL_DIR / "ann.keras",
        compile=False
    )

    lstm = tf.keras.models.load_model(
        MODEL_DIR / "lstm.keras",
        compile=False
    )

    with open(
        MODEL_DIR / "vocab.json",
        "r",
        encoding="utf-8"
    ) as f:
        vocabulary = json.load(f)

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_WORDS,
        output_mode="int",
        output_sequence_length=MAX_LENGTH,
        vocabulary=vocabulary
    )

    results = pd.read_csv(
        MODEL_DIR / "results.csv"
    )

    # Threshold selected during validation
    thresholds = dict(
        zip(
            results["Model"],
            results["Threshold"]
        )
    )

    return ann, lstm, vectorizer, results, thresholds


try:
    ann, lstm, vectorizer, results, thresholds = (
        load_resources()
    )

except Exception as e:
    st.error(
        "Could not load the trained models. "
        "Please run python train.py first."
    )
    st.exception(e)
    st.stop()

# ==========================================
# 4. HELPERS
# ==========================================

def predict_message(message, model, threshold):

    sequence = vectorizer(
        tf.constant([message])
    )

    probability = float(
        model.predict(
            sequence,
            verbose=0
        )[0][0]
    )

    prediction = (
        "SPAM"
        if probability >= threshold
        else "HAM"
    )

    return prediction, probability


def meter_html(probability, threshold):
    p = min(1.0, max(0.0, probability)) * 100
    t = min(1.0, max(0.0, threshold)) * 100
    return (
        '<div class="meter">'
        f'<div class="thr" style="left:{t:.1f}%"></div>'
        f'<div class="marker" style="left:calc({p:.1f}% - 2px)"></div>'
        '</div>'
        '<div class="meter-labels"><span>0% · Safe</span>'
        f'<span>Threshold {t:.0f}%</span><span>Spam · 100%</span></div>'
    )


def set_sample(text):
    st.session_state["message_input"] = text


def clear_message():
    st.session_state["message_input"] = ""


# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("## 🛡️ SpamShield AI")
    st.caption("ANN vs LSTM · SMS Classification")
    st.divider()

    model_choice = st.radio(
        "Primary model",
        ["LSTM", "ANN"],
        help="LSTM reads word order; ANN looks at the message as a whole."
    )

    st.markdown("#### Thresholds")
    for name in ["LSTM", "ANN"]:
        st.progress(
            min(1.0, max(0.0, float(thresholds[name]))),
            text=f"{name}: {float(thresholds[name]) * 100:.0f}%"
        )
    st.caption("Chosen on validation data to balance precision and recall.")

    st.divider()
    st.markdown("#### Try an example")
    for label, text in SAMPLES.items():
        st.button(
            label,
            key=f"sample_{label}",
            on_click=set_sample,
            args=(text,),
            use_container_width=True,
        )

# ==========================================
# 6. HERO
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🛡️ SpamShield AI</h1>
    <p>Paste any SMS and let two deep-learning models decide if it's spam or legitimate.</p>
    <span class="pill">🧠 ANN</span>
    <span class="pill">🔁 LSTM</span>
    <span class="pill">⚡ Real-time</span>
    <span class="pill">🎯 Validation-tuned threshold</span>
</div>
""", unsafe_allow_html=True)

tab_analyze, tab_perf, tab_about = st.tabs(
    ["🔍 Analyze", "📊 Performance", "ℹ️ How it works"]
)

# ==========================================
# 7. ANALYZE TAB
# ==========================================

with tab_analyze:

    message = st.text_area(
        "Enter your SMS message",
        key="message_input",
        placeholder=(
            "Example: Congratulations! "
            "You have won a lottery prize!"
        ),
        height=150,
        label_visibility="collapsed",
    )

    words = len(message.split())
    st.caption(f"{len(message)} characters · {words} words")

    col_a, col_b = st.columns([4, 1])
    with col_a:
        analyze = st.button(
            "🔍 Analyze Message",
            type="primary",
            use_container_width=True
        )
    with col_b:
        st.button(
            "Clear",
            on_click=clear_message,
            use_container_width=True
        )

    if analyze:

        if not message.strip():
            st.warning(
                "Please enter a message before analyzing."
            )

        else:

            selected_model = (
                lstm if model_choice == "LSTM" else ann
            )

            threshold = float(
                thresholds[model_choice]
            )

            with st.spinner("Analyzing message..."):
                prediction, probability = predict_message(
                    message,
                    selected_model,
                    threshold
                )

                ann_result, ann_prob = predict_message(
                    message,
                    ann,
                    float(thresholds["ANN"])
                )

                lstm_result, lstm_prob = predict_message(
                    message,
                    lstm,
                    float(thresholds["LSTM"])
                )

            st.markdown("### Detection Result")

            if prediction == "SPAM":
                st.markdown(
                    '<div class="verdict spam">'
                    '<div class="icon">🚨</div>'
                    '<div><div class="label">Spam detected</div>'
                    f'<div class="sub">{model_choice} flagged this message as suspicious.</div></div>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="verdict ham">'
                    '<div class="icon">✅</div>'
                    '<div><div class="label">Looks legitimate (Ham)</div>'
                    f'<div class="sub">{model_choice} found no strong spam signals.</div></div>'
                    '</div>',
                    unsafe_allow_html=True
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Spam Probability", f"{probability * 100:.2f}%")
            c2.metric("Decision Threshold", f"{threshold * 100:.0f}%")
            c3.metric("Model Used", model_choice)

            st.markdown(
                '<div class="card"><div class="card-title">Spam score</div>'
                + meter_html(probability, threshold)
                + '</div>',
                unsafe_allow_html=True
            )
            st.caption(
                "The percentage is the model's estimated "
                "spam probability, not a guarantee."
            )

            # ---- ANN vs LSTM ----
            st.markdown("### ⚖️ ANN vs LSTM — Same Message")

            col1, col2 = st.columns(2)

            for col, name, res, prob, thr in [
                (col1, "ANN", ann_result, ann_prob, thresholds["ANN"]),
                (col2, "LSTM", lstm_result, lstm_prob, thresholds["LSTM"]),
            ]:
                with col:
                    css = "spam" if res == "SPAM" else "ham"
                    icon = "🚨 SPAM" if res == "SPAM" else "✅ HAM"
                    st.markdown(
                        '<div class="card">'
                        f'<div class="card-title">{name} prediction</div>'
                        f'<span class="tag {css}">{icon}</span>'
                        f'<div class="big-number" style="margin-top:10px">{prob * 100:.2f}%</div>'
                        '<div class="card-title" style="margin-top:2px">spam probability</div>'
                        + meter_html(prob, float(thr))
                        + '</div>',
                        unsafe_allow_html=True
                    )

            if ann_result != lstm_result:
                st.info(
                    "The models disagree on this message. "
                    "This can happen because ANN and LSTM "
                    "learn different text patterns."
                )
            else:
                st.success("Both models agree on this message.")

# ==========================================
# 8. PERFORMANCE TAB
# ==========================================

with tab_perf:

    st.markdown("### Model Performance Comparison")

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    best = results.loc[results["F1 Score"].idxmax()]
    st.markdown(
        '<div class="card">'
        '<div class="card-title">Best F1 score</div>'
        f'<div class="big-number">{best["Model"]} · {best["F1 Score"] * 100:.2f}%</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.write("")

    cols = st.columns(len(results))
    for col, (_, row) in zip(cols, results.iterrows()):
        with col:
            st.markdown(f"**{row['Model']}**")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{row['Accuracy'] * 100:.2f}%")
            m2.metric("F1 Score", f"{row['F1 Score'] * 100:.2f}%")
            m3, m4 = st.columns(2)
            m3.metric("Precision", f"{row['Precision'] * 100:.2f}%")
            m4.metric("Recall", f"{row['Recall'] * 100:.2f}%")

    st.markdown("#### Detailed table")

    display_results = results.copy()
    for metric in metrics:
        display_results[metric] = (
            display_results[metric] * 100
        ).round(2)

    st.dataframe(
        display_results,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("#### Performance chart")

    chart_data = results.set_index("Model")[metrics] * 100
    st.bar_chart(chart_data)

# ==========================================
# 9. ABOUT TAB
# ==========================================

with tab_about:

    st.markdown("### How does this project work?")

    steps = [
        ("Input", "The user enters an SMS message."),
        ("Vectorize", "TextVectorization converts the message into numerical sequences."),
        ("Embed", "The Embedding layer turns word IDs into trainable numerical representations."),
        ("Model", "The ANN or LSTM processes the sequence."),
        ("Score", "A sigmoid output produces a spam probability."),
        ("Threshold", "The score is compared with the validation-selected threshold."),
        ("Verdict", "The message is classified as Spam or Ham."),
    ]

    html = '<div class="card">'
    for i, (title, desc) in enumerate(steps, 1):
        html += (
            f'<div class="step"><div class="num">{i}</div>'
            f'<div class="txt"><b>{title}</b><br><span>{desc}</span></div></div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

st.write("")
st.caption(
    "Deep Learning Mini Project | "
    "ANN vs LSTM SMS Spam Classification"
)
