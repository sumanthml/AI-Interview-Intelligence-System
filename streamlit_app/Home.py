import sys
from pathlib import Path

# Add project root to sys.path so 'src' is always discoverable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config.config import (
    DATASET_NAME,
    DEVICE,
    MAX_SEQUENCE_LENGTH,
    MODEL_NAME,
    NUM_CLASSES,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
)

st.set_page_config(
    page_title="Amazon Review Intelligence Platform",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        margin: 4px;
        background-color: #e0e7ff;
        color: #3730a3;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .feature-box {
        border-left: 4px solid #3b82f6;
        background-color: #f0fdf4;
        padding: 15px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🛍️ Amazon Review Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade NLP pipeline & Explainable AI (XAI) for fine-grained customer sentiment analysis.</div>', unsafe_allow_html=True)

# Top KPI metrics
col1, col2, col3, col4 = st.columns(4)

dataset_rows = 568454
if PROCESSED_DATA_FILE.exists():
    try:
        df_sample = pd.read_csv(PROCESSED_DATA_FILE, nrows=10)
        # Approximate size
        dataset_rows = 393579
    except Exception:
        pass

with col1:
    st.markdown(
        """
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color:#64748b;">Base Architecture</h3>
            <h2 style="margin:5px 0 0 0; color:#0f172a;">DistilBERT</h2>
            <span style="font-size:0.8rem; color:#10b981;">Transformer Backbone</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color:#64748b;">Classification Classes</h3>
            <h2 style="margin:5px 0 0 0; color:#0f172a;">{NUM_CLASSES} Classes</h2>
            <span style="font-size:0.8rem; color:#3b82f6;">Pos / Neu / Neg</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color:#64748b;">Compute Device</h3>
            <h2 style="margin:5px 0 0 0; color:#0f172a;">{DEVICE.upper()}</h2>
            <span style="font-size:0.8rem; color:#8b5cf6;">Hardware Acceleration</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <h3 style="margin:0; font-size: 1rem; color:#64748b;">Max Sequence Length</h3>
            <h2 style="margin:5px 0 0 0; color:#0f172a;">{MAX_SEQUENCE_LENGTH} Tokens</h2>
            <span style="font-size:0.8rem; color:#f59e0b;">Subword Attention</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Architecture & Modules Section
c1, c2 = st.columns([3, 2])

with c1:
    st.subheader("📌 System Architecture & Pipeline")
    st.markdown(
        """
        This platform delivers an **end-to-end Machine Learning Lifecycle (MLOps)** for e-commerce sentiment intelligence:
        
        1. **Data Ingestion & Cleaning**: Ingests massive Amazon Fine Food review corpora, removes HTML artifacts, strips URLs, normalizes Unicode, deduplicates text, and stratifies into Train / Validation / Test sets.
        2. **Transformer Fine-Tuning**: PyTorch & Hugging Face `distilbert-base-uncased` fine-tuned with AdamW, linear learning rate warm-up, and Cross-Entropy loss.
        3. **Explainable AI (XAI)**: Token-level integrated attributions and gradient-based importance scoring highlight influential keywords behind every prediction.
        4. **Production Web Interface**: Multi-page Streamlit dashboard providing live interactive predictions, batch CSV scoring, exploratory analytics, and model evaluation metrics.
        """
    )
    
    st.info("💡 **Quick Navigation:** Use the left sidebar to navigate across **Live Prediction**, **Dataset Analytics**, **Model Performance**, and **Model Explainability**!")

with c2:
    st.subheader("🛠️ Technology Stack")
    badges = [
        "PyTorch", "Transformers", "DistilBERT", "Pandas", "NumPy",
        "Scikit-Learn", "Streamlit", "Plotly", "Matplotlib", "Optuna",
        "Tokenizers", "WordCloud", "Accelerate", "Python 3"
    ]
    st.markdown("".join([f'<span class="badge-pill">{b}</span>' for b in badges]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Dataset Summary")
    st.write(f"- **Corpus**: {DATASET_NAME}")
    st.write(f"- **Target Task**: Multi-Class Sentiment Analysis")
    st.write(f"- **Labels**: 1-2 Stars (Negative) | 3 Stars (Neutral) | 4-5 Stars (Positive)")

st.markdown("---")
st.markdown("Developed with ❤️ for Advanced NLP Engineering & Review Intelligence.")
