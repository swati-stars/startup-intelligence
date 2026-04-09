# dashboards/pmf_dashboard.py
# PURPOSE: Visual PMF dashboard — run with: streamlit run dashboards/pmf_dashboard.py
#
# This is what founders/investors will see.
# Design principle: every number should answer a business question.

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.scraper import collect_data
from src.nlp_pipeline import run_pipeline
from src.pmf_scorer import calculate_pmf_score

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PMF Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.title("🎯 Product–Market Fit Analyzer")
st.markdown("*Predict whether a product has genuine market demand using NLP on user reviews*")

# ─── SIDEBAR: DATA INPUT ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Analyze a product")

    app_id = st.text_input(
        "Google Play App ID",
        value="com.notion.id",
        help="The package name from Play Store URL"
    )
    app_name = st.text_input("App Name", value="Notion")

    # Option to use cached data (faster) or re-scrape
    use_cache = st.checkbox("Use cached data (faster)", value=True)

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_process(app_id, app_name):
    raw_path = f"data/{app_name.lower()}_raw.csv"
    processed_path = f"data/{app_name.lower()}_processed.csv"

    # Use cached processed data if it exists
    if os.path.exists(processed_path):
        return pd.read_csv(processed_path)

    # Use cached raw data if it exists
    if os.path.exists(raw_path):
        with st.spinner("Running NLP analysis..."):
            run_pipeline(raw_path, processed_path)
        return pd.read_csv(processed_path)

    # Otherwise scrape fresh
    with st.spinner("Scraping reviews... this takes ~30 seconds"):
        try:
            collect_data(app_id, app_name, reddit_query=app_name)
            run_pipeline(raw_path, processed_path)
            return pd.read_csv(processed_path)
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            return None


# ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
if analyze_btn or os.path.exists(f"data/notion_processed.csv"):

    # Load demo data if no specific app requested
    app_name_clean = app_name if analyze_btn else "notion"

    try:
        df = load_and_process(app_id, app_name_clean)
        scores = calculate_pmf_score(df)

        # ── PMF SCORE GAUGE ──────────────────────────────────────────────────
        st.subheader("PMF Score")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Overall PMF Score", f"{scores['pmf_score']}/100")
        with col2:
            st.metric("Sentiment Score", f"{scores['sentiment_score']}/100")
        with col3:
            st.metric("Engagement Score", f"{scores['engagement_score']}/100")
        with col4:
            st.metric("Pain Intensity", f"{scores['pain_intensity_score']}/100")
        with col5:
            st.metric("Feature Demand", f"{scores['feature_demand_score']}/100")

        # Gauge chart for overall PMF score
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=scores['pmf_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 35], 'color': "#ff4444"},
                    {'range': [35, 55], 'color': "#ffaa00"},
                    {'range': [55, 75], 'color': "#88cc00"},
                    {'range': [75, 100], 'color': "#00cc44"},
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': scores['pmf_score']}
            },
            title={'text': "PMF Score"}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ── INTERPRETATION ───────────────────────────────────────────────────
        st.info(f"**Interpretation:** {scores['interpretation']}")

        # ── BUSINESS RECOMMENDATION ──────────────────────────────────────────
        st.subheader("📋 Business Recommendation")
        st.success(scores['recommendation'])

        # ── SENTIMENT BREAKDOWN ──────────────────────────────────────────────
        st.subheader("Sentiment Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            sentiment_counts = df['sentiment_label'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Review Sentiment",
                color=sentiment_counts.index,
                color_discrete_map={
                    'positive': '#00cc44',
                    'neutral': '#aaaaaa',
                    'negative': '#ff4444'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Sentiment over time — shows if product is improving or declining
            if 'at' in df.columns:
                df['at'] = pd.to_datetime(df['at'])
                df['month'] = df['at'].dt.to_period('M').astype(str)
                monthly = df.groupby(['month', 'sentiment_label']).size().unstack(fill_value=0)

                fig_time = px.line(
                    monthly,
                    title="Sentiment Trend Over Time",
                    labels={'value': 'Review Count', 'month': 'Month'}
                )
                st.plotly_chart(fig_time, use_container_width=True)

        # ── TOPIC DISTRIBUTION ───────────────────────────────────────────────
        st.subheader("What Users Are Talking About")

        topic_sentiment = df.groupby(['topic_category', 'sentiment_label']).size().unstack(fill_value=0)
        fig_bar = px.bar(
            topic_sentiment,
            barmode='group',
            title="Topics by Sentiment",
            color_discrete_map={
                'positive': '#00cc44',
                'neutral': '#aaaaaa',
                'negative': '#ff4444'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── RAW DATA EXPLORER ─────────────────────────────────────────────────
        with st.expander("View raw data"):
            st.dataframe(
                df[['text', 'sentiment_label', 'sentiment_compound', 'topic_category']].head(50),
                use_container_width=True
            )

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Run the scraper first: `python src/scraper.py`")

else:
    st.info("Enter an app ID in the sidebar and click Analyze to begin.")
