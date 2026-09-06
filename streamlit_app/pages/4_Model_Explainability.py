import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explainability.shap_analysis import ModelExplainer

st.set_page_config(page_title="Model Explainability (XAI)", page_icon="🔍", layout="wide")

st.title("🔍 Model Explainability & Token Attributions")
st.markdown("Understand **why** the Transformer model made a specific sentiment prediction using integrated gradient feature attributions.")

@st.cache_resource
def get_explainer():
    return ModelExplainer()

explainer = get_explainer()

st.subheader("1. Interactive Review Explainer")
st.markdown("Select an interesting edge case or enter your own custom text:")

demo_cases = {
    "🌟 Strong Positive": "These gourmet chocolates are outstanding, rich and arrived in pristine condition!",
    "🔴 Strong Negative": "Terrible quality. Rancid smell, expired two months ago, and customer service was rude.",
    "🔄 Negation Edge Case": "Not bad at all, actually exceeded my expectations given the low price.",
    "⚖️ Mixed Review": "The flavor is truly fantastic, but the packaging arrived completely crushed and ruined.",
}

selected_case = st.selectbox("Choose a preset scenario:", list(demo_cases.keys()))
custom_text = st.text_area("Review text for explanation:", value=demo_cases[selected_case], height=100)

if st.button("🔎 Explain Predictions & Word Importance", type="primary"):
    if not custom_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Computing token attributions and gradients..."):
            res = explainer.explain_text(custom_text)
            pred = res.get("full_prediction", {})
            sentiment = res["predicted_sentiment"]
            tokens = res["tokens"]
            scores = res["attributions"]
            top_tokens = res["top_influential_tokens"]

            st.markdown("---")
            col_pred1, col_pred2 = st.columns([1, 2])

            with col_pred1:
                st.markdown("### 🎯 Predicted Sentiment")
                if sentiment == "Positive":
                    st.success(f"### 🟢 **{sentiment}**")
                elif sentiment == "Negative":
                    st.error(f"### 🔴 **{sentiment}**")
                else:
                    st.warning(f"### 🟡 **{sentiment}**")
                
                if "confidence" in pred:
                    st.metric("Confidence", f"{pred['confidence']*100:.1f}%")

            with col_pred2:
                st.markdown("### 🎨 Token Attribution Map")
                st.markdown(res["html_visualization"], unsafe_allow_html=True)
                st.caption("🟢 **Green tokens** drive sentiment toward **Positive** | 🔴 **Red tokens** drive sentiment toward **Negative**.")

            st.markdown("---")
            st.subheader("2. Keyword Impact Ranking")

            if top_tokens:
                token_df = pd.DataFrame(top_tokens, columns=["Token", "Attribution Score"])
                token_df["Impact Type"] = token_df["Attribution Score"].apply(
                    lambda x: "Positive Influence" if x > 0 else "Negative Influence"
                )
                token_df = token_df.sort_values(by="Attribution Score", ascending=True)

                fig_bar = go.Figure(
                    go.Bar(
                        x=token_df["Attribution Score"],
                        y=token_df["Token"],
                        orientation="h",
                        marker=dict(
                            color=["#ef4444" if s < 0 else "#22c55e" for s in token_df["Attribution Score"]],
                        ),
                        text=[f"{s:+.3f}" for s in token_df["Attribution Score"]],
                        textposition="auto",
                    )
                )
                fig_bar.update_layout(
                    title="Top Contributing Tokens (Signed Feature Impact)",
                    xaxis_title="Attribution Score (-1.0 to +1.0)",
                    yaxis_title="Word / Subword Token",
                    height=400,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with st.expander("📋 View Complete Token Attribution Data Table"):
                full_table = pd.DataFrame({"Token": tokens, "Attribution Score": scores})
                st.dataframe(full_table, use_container_width=True)

st.markdown("---")
st.subheader("3. Technical Architecture of Explainability")
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.markdown("#### 🧠 How Token Attribution Works")
    st.markdown(
        """
        1. **Embedding Layer Interception**: We hook into DistilBERT's word embedding layer to capture input embeddings $E$.
        2. **Backpropagation of Output Logits**: We compute the gradient $\\nabla_E y_{target}$ of the predicted class with respect to embedding representations.
        3. **Input $\\times$ Gradient Metric**: $Attribution_i = E_i \\cdot \\nabla_{E_i} y_{target}$ measures how much token $i$ pushed the model toward the prediction.
        """
    )

with exp_col2:
    st.markdown("#### 🛡️ Value in Business & E-Commerce")
    st.markdown(
        """
        - **Model Trust & Governance**: Validates that the neural network focuses on genuine semantic cues rather than spurious artifacts.
        - **Aspect Extraction**: Pinpoints specific product qualities (e.g. *flavor*, *packaging*, *smell*) linked to negative or positive customer experiences.
        - **Auditing Edge Cases**: Reveals how negations like *not bad* or mixed reviews are resolved by transformer self-attention layers.
        """
    )
