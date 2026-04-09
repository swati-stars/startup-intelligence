# dashboards/feedback_dashboard.py
# PURPOSE: Convert messy reviews into actionable feedback categories
# Run with: streamlit run dashboards/feedback_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(page_title="Feedback Intelligence", page_icon="📊", layout="wide")

st.title("📊 Customer Feedback Intelligence Dashboard")
st.markdown("*Transform messy app reviews into structured business insights*")

with st.sidebar:
    st.header("Settings")
    
    # Auto-detect all processed CSV files in data/ folder
    data_folder = "data"
    os.makedirs(data_folder, exist_ok=True)
    
    available_files = [
        f for f in os.listdir(data_folder) 
        if f.endswith("_processed.csv")
    ]
    
    if not available_files:
        st.warning("No processed data found yet.")
        st.info("Analyze an app in the PMF Analyzer first, then come back here.")
        st.stop()
    
    # Clean display names — "notion_processed.csv" → "Notion"
    display_names = {
        f: f.replace("_processed.csv", "").replace("_", " ").title() 
        for f in available_files
    }
    
    selected_display = st.selectbox(
        "Select app to analyze",
        options=list(display_names.values())
    )
    
    # Get actual filename from display name
    data_file = [k for k, v in display_names.items() if v == selected_display][0]


# Load data
@st.cache_data
def load_data(filename):
    return pd.read_csv(f"data/{filename}")

df = load_data(data_file)

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
    st.subheader("Executive Summary")

    total = len(df)
    neg_pct = (df['sentiment_label'] == 'negative').mean() * 100
    top_complaint_cat = df[df['sentiment_label'] == 'negative']['topic_category'].mode()[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analyzed", f"{total:,}")
    col2.metric("Negative Feedback Rate", f"{neg_pct:.1f}%")
    col3.metric("Top Complaint Category", top_complaint_cat)
    col4.metric("Avg Sentiment Score", f"{df['sentiment_compound'].mean():.2f}")

    # Business insight banner
    st.warning(
        f"**Key Finding:** {neg_pct:.0f}% of users report negative experiences. "
        f"The primary driver is **{top_complaint_cat}**, affecting {(df['topic_category'] == top_complaint_cat).mean() * 100:.0f}% "
        f"of all reviews. Fixing this could directly improve retention."
    )

    # ── COMPLAINT BREAKDOWN TABLE ──────────────────────────────────────────────
    st.subheader("Complaint Category Breakdown")

    # Build the summary table that investors/founders love
    category_stats = []
    for cat in df['topic_category'].unique():
        cat_df = df[df['topic_category'] == cat]
        neg_in_cat = (cat_df['sentiment_label'] == 'negative').mean() * 100
        total_in_cat = len(cat_df)
        pct_of_all = (total_in_cat / total) * 100
        avg_sentiment = cat_df['sentiment_compound'].mean()

        category_stats.append({
            'Category': cat,
            '% of All Reviews': f"{pct_of_all:.1f}%",
            'Negative Rate': f"{neg_in_cat:.1f}%",
            'Avg Sentiment': round(avg_sentiment, 2),
            'Review Count': total_in_cat,
            'Priority': '🔴 High' if neg_in_cat > 60 else ('🟡 Medium' if neg_in_cat > 30 else '🟢 Low')
        })

    stats_df = pd.DataFrame(category_stats).sort_values('Review Count', ascending=False)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # ── VISUAL BREAKDOWNS ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Treemap: shows category SIZE by proportion
        fig_treemap = px.treemap(
            df,
            path=['topic_category', 'sentiment_label'],
            title="Feedback Volume by Category",
            color='sentiment_compound',
            color_continuous_scale='RdYlGn',
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

    with col2:
        # Horizontal bar: % of users complaining per category
        complaint_pcts = (
                df[df['sentiment_label'] == 'negative']
                .groupby('topic_category').size()
                .div(total) * 100
        ).sort_values(ascending=True)

        fig_bar = px.bar(
            x=complaint_pcts.values,
            y=complaint_pcts.index,
            orientation='h',
            title="% of All Users Reporting Issues Per Category",
            labels={'x': '% of total users', 'y': 'Category'},
            color=complaint_pcts.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── SAMPLE REVIEWS PER CATEGORY ───────────────────────────────────────────
    st.subheader("Dive into a Category")

    selected_cat = st.selectbox("Choose category", df['topic_category'].unique())
    selected_sentiment = st.radio("Sentiment filter", ['negative', 'positive', 'all'], horizontal=True)

    filtered = df[df['topic_category'] == selected_cat]
    if selected_sentiment != 'all':
        filtered = filtered[filtered['sentiment_label'] == selected_sentiment]

    # Show top reviews sorted by engagement (length = more detailed = more useful)
    filtered = filtered.copy()
    filtered['length'] = filtered['text'].str.len()
    sample = filtered.nlargest(10, 'length')[['text', 'sentiment_label', 'sentiment_compound']]

    for _, row in sample.iterrows():
        color = "🟢" if row['sentiment_label'] == 'positive' else "🔴" if row['sentiment_label'] == 'negative' else "⚪"
        st.markdown(f"{color} *\"{row['text'][:300]}...\"*")
        st.caption(f"Sentiment score: {row['sentiment_compound']:.2f}")
        st.divider()

else:
    st.info("No processed data found. Run the scraper and NLP pipeline first.")
    st.code("python src/scraper.py\npython src/nlp_pipeline.py")
