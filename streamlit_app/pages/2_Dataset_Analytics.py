import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud

from src.config.config import PROCESSED_DATA_FILE, RAW_DATA_FILE

st.set_page_config(page_title="Dataset Analytics", page_icon="📊", layout="wide")

st.title("📊 Dataset Analytics & Exploratory Insights")
st.markdown("Deep dive into customer sentiment distribution, review lengths, and rating patterns across the Amazon corpus.")

@st.cache_data
def load_eda_data():
    if PROCESSED_DATA_FILE.exists():
        df = pd.read_csv(PROCESSED_DATA_FILE, nrows=50000)
    elif RAW_DATA_FILE.exists():
        df = pd.read_csv(RAW_DATA_FILE, nrows=50000)
        df["sentiment"] = df["Score"].map({1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"})
        df["review"] = df["Summary"].fillna("").astype(str) + ". " + df["Text"].fillna("").astype(str)
    else:
        # Fallback synthetic demo data
        data = {
            "review": [
                "Delicious coffee, very smooth", "Awful taste, threw it away",
                "Decent product, average quality", "Best chocolate ever!",
                "Stale and expired on arrival", "Not bad, but expensive",
                "Super fresh and tasty", "Never buying again!",
                "Great value for money", "Neutral opinion, standard taste"
            ] * 500,
            "Score": [5, 1, 3, 5, 1, 3, 5, 1, 4, 3] * 500,
            "sentiment": ["Positive", "Negative", "Neutral", "Positive", "Negative", "Neutral", "Positive", "Negative", "Positive", "Neutral"] * 500,
        }
        df = pd.DataFrame(data)
    df["review_length"] = df["review"].astype(str).str.len()
    df["word_count"] = df["review"].astype(str).apply(lambda x: len(x.split()))
    return df

with st.spinner("Loading dataset telemetry..."):
    df = load_eda_data()

# KPI Metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Sampled Reviews", f"{len(df):,}")
with kpi2:
    pos_pct = (df["sentiment"] == "Positive").mean() * 100
    st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
with kpi3:
    neg_pct = (df["sentiment"] == "Negative").mean() * 100
    st.metric("Negative Sentiment", f"{neg_pct:.1f}%")
with kpi4:
    avg_len = df["review_length"].median()
    st.metric("Median Review Length", f"{int(avg_len)} chars")

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("1. Sentiment Class Distribution")
    s_counts = df["sentiment"].value_counts().reset_index()
    s_counts.columns = ["Sentiment", "Count"]

    bar_fig = px.bar(
        s_counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#eab308"},
        text="Count",
        title="Volume by Sentiment Category",
    )
    bar_fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    bar_fig.update_layout(height=380)
    st.plotly_chart(bar_fig, use_container_width=True)

with col_right:
    st.subheader("2. Star Rating (1 to 5) Breakdown")
    if "Score" in df.columns:
        r_counts = df["Score"].value_counts().sort_index().reset_index()
        r_counts.columns = ["Star Rating", "Count"]
        r_counts["Star Rating"] = r_counts["Star Rating"].astype(str) + " ⭐"

        donut_fig = px.pie(
            r_counts,
            names="Star Rating",
            values="Count",
            hole=0.45,
            title="Star Rating Proportions",
            color_discrete_sequence=px.colors.sequential.Tealgrn,
        )
        donut_fig.update_layout(height=380)
        st.plotly_chart(donut_fig, use_container_width=True)

st.markdown("---")

c_len1, c_len2 = st.columns(2)

with c_len1:
    st.subheader("3. Review Length vs. Sentiment")
    box_fig = px.box(
        df[df["review_length"] <= df["review_length"].quantile(0.98)],
        x="sentiment",
        y="review_length",
        color="sentiment",
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#eab308"},
        title="Character Length Distribution by Class",
        labels={"review_length": "Length (Characters)", "sentiment": "Sentiment"},
    )
    box_fig.update_layout(height=380)
    st.plotly_chart(box_fig, use_container_width=True)

with c_len2:
    st.subheader("4. Word Count Density Histogram")
    hist_fig = px.histogram(
        df[df["word_count"] <= df["word_count"].quantile(0.98)],
        x="word_count",
        color="sentiment",
        barmode="overlay",
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#eab308"},
        title="Word Count Frequency Distribution",
        labels={"word_count": "Word Count"},
    )
    hist_fig.update_layout(height=380)
    st.plotly_chart(hist_fig, use_container_width=True)

st.markdown("---")

st.subheader("5. Lexical Word Clouds")
wc_col1, wc_col2 = st.columns(2)

with wc_col1:
    st.markdown("#### 🟢 Positive Lexicon")
    pos_texts = " ".join(df[df["sentiment"] == "Positive"]["review"].dropna().head(2000).tolist())
    if pos_texts:
        wc_pos = WordCloud(width=600, height=300, background_color="white", colormap="Greens", max_words=75).generate(pos_texts)
        fig_wc1, ax_wc1 = plt.subplots(figsize=(8, 4))
        ax_wc1.imshow(wc_pos, interpolation="bilinear")
        ax_wc1.axis("off")
        st.pyplot(fig_wc1)

with wc_col2:
    st.markdown("#### 🔴 Negative Lexicon")
    neg_texts = " ".join(df[df["sentiment"] == "Negative"]["review"].dropna().head(2000).tolist())
    if neg_texts:
        wc_neg = WordCloud(width=600, height=300, background_color="white", colormap="Reds", max_words=75).generate(neg_texts)
        fig_wc2, ax_wc2 = plt.subplots(figsize=(8, 4))
        ax_wc2.imshow(wc_neg, interpolation="bilinear")
        ax_wc2.axis("off")
        st.pyplot(fig_wc2)

st.markdown("---")
st.subheader("6. Interactive Dataset Explorer")
filter_sentiment = st.multiselect("Filter by Sentiment:", options=["Positive", "Neutral", "Negative"], default=["Positive", "Negative"])
search_term = st.text_input("Search reviews for keyword:")

filtered_df = df[df["sentiment"].isin(filter_sentiment)]
if search_term.strip():
    filtered_df = filtered_df[filtered_df["review"].astype(str).str.contains(search_term, case=False, na=False)]

st.dataframe(filtered_df[["review", "Score", "sentiment", "word_count"]].head(100), use_container_width=True, height=280)
