import io
import sys
import time
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
from src.inference.predictor import SentimentPredictor

st.set_page_config(page_title="Live Sentiment Prediction", page_icon="⚡", layout="wide")

st.title("⚡ Real-Time Sentiment Prediction")
st.markdown("Test single reviews or process batches of customer reviews with the fine-tuned Transformer model.")

@st.cache_resource
def load_predictor():
    return SentimentPredictor()

@st.cache_resource
def load_explainer():
    return ModelExplainer()

predictor = load_predictor()
explainer = load_explainer()

tab1, tab2 = st.tabs(["💬 Single Review Analysis", "📁 Batch CSV Processing"])

with tab1:
    st.subheader("Interactive Review Analyzer")

    # Sample review buttons for quick demo
    st.markdown("**Try a sample review or type your own:**")
    sample_col1, sample_col2, sample_col3 = st.columns(3)
    
    sample_text = ""
    if sample_col1.button("🌟 Positive Sample", use_container_width=True):
        sample_text = "This organic coffee is absolute perfection! Rich aroma, velvety smooth finish, and arrived fast."
    if sample_col2.button("⚠️ Neutral Sample", use_container_width=True):
        sample_text = "The product is okay for the price. Quality is average and packaging was standard."
    if sample_col3.button("❌ Negative Sample", use_container_width=True):
        sample_text = "Terrible experience. The package arrived completely smashed, leaking everywhere, and customer support refused a refund."

    user_input = st.text_area(
        "Enter customer review text:",
        value=sample_text if sample_text else "The flavor is delicious and authentic! Easily my favorite pantry purchase this year.",
        height=120,
        placeholder="Type or paste your review here...",
    )

    if st.button("🚀 Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some review text first.")
        else:
            with st.spinner("Classifying sentiment with DistilBERT..."):
                res = predictor.predict(user_input)
                sentiment = res["sentiment"]
                confidence = res["confidence"]
                probs = res["probabilities"]
                latency = res["latency_ms"]

                st.markdown("---")
                # Results layout
                res_col1, res_col2 = st.columns([1, 1.2])

                with res_col1:
                    st.markdown("### Prediction Result")
                    if sentiment == "Positive":
                        st.success(f"### 🟢 **{sentiment.upper()}**")
                    elif sentiment == "Negative":
                        st.error(f"### 🔴 **{sentiment.upper()}**")
                    else:
                        st.warning(f"### 🟡 **{sentiment.upper()}**")

                    st.metric("Model Confidence", f"{confidence * 100:.1f}%")
                    st.caption(f"⏱️ Inference Latency: **{latency:.1f} ms**")

                with res_col2:
                    st.markdown("### Class Probabilities")
                    prob_df = pd.DataFrame(
                        {
                            "Sentiment": list(probs.keys()),
                            "Probability": [v * 100 for v in probs.values()],
                            "Color": ["#e74c3c", "#f39c12", "#2ecc71"],
                        }
                    )

                    fig = go.Figure(
                        go.Bar(
                            x=prob_df["Probability"],
                            y=prob_df["Sentiment"],
                            orientation="h",
                            marker=dict(
                                color=["#ef4444" if s == "Negative" else "#eab308" if s == "Neutral" else "#22c55e" for s in prob_df["Sentiment"]],
                            ),
                            text=[f"{p:.1f}%" for p in prob_df["Probability"]],
                            textposition="auto",
                        )
                    )
                    fig.update_layout(
                        xaxis_title="Probability (%)",
                        yaxis_title="",
                        height=200,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(range=[0, 100]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Token attribution highlight
                st.markdown("### 🔍 Token-Level Influence Preview")
                exp_res = explainer.explain_text(user_input)
                st.markdown(exp_res["html_visualization"], unsafe_allow_html=True)
                st.caption("🟢 Green highlights indicate words supporting positive sentiment, 🔴 Red highlights indicate negative push.")

with tab2:
    st.subheader("Batch File Processing")
    st.markdown("Upload a CSV file containing reviews to run sentiment inference at scale.")

    uploaded_file = st.file_uploader("Upload CSV (must contain a text column)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded `{uploaded_file.name}` with **{len(batch_df):,}** rows.")

            # Column selector
            text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
            default_col = "review" if "review" in text_cols else ("Text" if "Text" in text_cols else text_cols[0] if text_cols else None)

            if not text_cols:
                st.error("No text column found in CSV.")
            else:
                selected_col = st.selectbox("Select Review Text Column:", text_cols, index=text_cols.index(default_col) if default_col in text_cols else 0)

                max_batch = st.slider("Limit reviews to process:", min_value=5, max_value=min(1000, len(batch_df)), value=min(50, len(batch_df)))

                if st.button("⚡ Process Batch", type="primary"):
                    subset_df = batch_df.iloc[:max_batch].copy()
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    sentiments = []
                    confidences = []

                    start_t = time.time()
                    for idx, row in enumerate(subset_df[selected_col].astype(str)):
                        pred = predictor.predict(row)
                        sentiments.append(pred["sentiment"])
                        confidences.append(round(pred["confidence"] * 100, 2))
                        progress_bar.progress((idx + 1) / max_batch)
                        status_text.text(f"Processed {idx + 1}/{max_batch} reviews...")

                    subset_df["Predicted_Sentiment"] = sentiments
                    subset_df["Confidence_%"] = confidences

                    total_elapsed = time.time() - start_t
                    status_text.success(f"Completed {max_batch} reviews in {total_elapsed:.2f}s ({max_batch/total_elapsed:.1f} reviews/sec)!")

                    # Summary chart
                    s_counts = subset_df["Predicted_Sentiment"].value_counts().reset_index()
                    s_counts.columns = ["Sentiment", "Count"]

                    col_chart, col_data = st.columns([1, 2])
                    with col_chart:
                        pie_fig = px.pie(
                            s_counts,
                            names="Sentiment",
                            values="Count",
                            title="Batch Sentiment Breakdown",
                            color="Sentiment",
                            color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#eab308"},
                        )
                        st.plotly_chart(pie_fig, use_container_width=True)

                    with col_data:
                        st.dataframe(subset_df[[selected_col, "Predicted_Sentiment", "Confidence_%"]], use_container_width=True, height=300)

                    # Export button
                    csv_buffer = io.StringIO()
                    subset_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_buffer.getvalue(),
                        file_name="sentiment_predictions.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
