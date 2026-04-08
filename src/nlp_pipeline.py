# src/nlp_pipeline.py
# PURPOSE: Transform raw review text into structured insights
#
# What this does in sequence:
# 1. Clean the text (remove noise)
# 2. Score sentiment (positive/negative/neutral)
# 3. Cluster topics (what are people actually talking about?)
# 4. Extract pain points and feature requests

import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
import warnings

warnings.filterwarnings('ignore')


# ─── TEXT CLEANING ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Remove noise before NLP processing.
    Raw reviews contain URLs, emojis, special chars — these confuse NLP models.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s\'-]', ' ', text)  # Keep letters, apostrophes
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text


# ─── SENTIMENT ANALYSIS ───────────────────────────────────────────────────────
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is the right tool here.

    WHY VADER over others:
    - Designed for social media / short reviews (exactly our use case)
    - Understands capitalization ("GREAT" vs "great"), punctuation ("!!!")
    - Fast — no model loading, rule-based
    - Free, no API key needed

    Returns compound score: -1 (very negative) to +1 (very positive)
    We also label each review as positive/neutral/negative for the dashboard.
    """
    print("Running sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()

    df = df.copy()
    df['clean_text'] = df['text'].apply(clean_text)

    # Get sentiment scores for each review
    scores = df['clean_text'].apply(lambda x: analyzer.polarity_scores(x))
    df['sentiment_compound'] = scores.apply(lambda x: x['compound'])
    df['sentiment_positive'] = scores.apply(lambda x: x['pos'])
    df['sentiment_negative'] = scores.apply(lambda x: x['neg'])

    # Label: VADER convention — compound > 0.05 = positive, < -0.05 = negative
    df['sentiment_label'] = df['sentiment_compound'].apply(
        lambda s: 'positive' if s >= 0.05 else ('negative' if s <= -0.05 else 'neutral')
    )

    print(f"  ✓ Sentiment breakdown: {df['sentiment_label'].value_counts().to_dict()}")
    return df


# ─── TOPIC CLUSTERING ─────────────────────────────────────────────────────────
def cluster_topics(df: pd.DataFrame, n_clusters: int = 6) -> pd.DataFrame:
    """
    Group reviews into topics automatically using TF-IDF + KMeans.

    WHY THIS APPROACH:
    - TF-IDF converts text to numbers (word importance scores)
    - KMeans groups similar reviews together
    - We then read the top words per cluster to name the topic

    This reveals: "34% of complaints are about pricing" without manually
    reading thousands of reviews.

    n_clusters: start with 5-8. Too few = vague groups. Too many = noise.
    """
    print(f"Clustering into {n_clusters} topics...")

    df = df.copy()
    texts = df['clean_text'].fillna('').tolist()

    # TF-IDF: converts each review into a vector of word importance scores
    # min_df=5: ignore words appearing in fewer than 5 reviews (typos, rare words)
    # max_df=0.8: ignore words in >80% of reviews (too common to be meaningful)
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),  # Also capture 2-word phrases like "customer service"
        stop_words='english'
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster_id'] = kmeans.fit_predict(tfidf_matrix)

    # Get top 10 keywords per cluster — these help us NAME each cluster
    cluster_labels = {}
    for cluster_num in range(n_clusters):
        center = kmeans.cluster_centers_[cluster_num]
        top_indices = center.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        cluster_labels[cluster_num] = top_words

    # Auto-name clusters based on their keywords
    # This maps keyword patterns to human-readable category names
    category_map = auto_name_clusters(cluster_labels)
    df['topic_category'] = df['cluster_id'].map(category_map)

    print(f"  ✓ Topic distribution: {df['topic_category'].value_counts().to_dict()}")
    return df, cluster_labels


def auto_name_clusters(cluster_keywords: dict) -> dict:
    """
    Automatically assign a business-readable name to each cluster.
    Matches keyword patterns to known complaint categories.
    This is a heuristic — works well for app reviews.
    """
    category_patterns = {
        'Pricing & value': ['price', 'cost', 'expensive', 'cheap', 'subscription', 'pay', 'free', 'money'],
        'Bugs & crashes': ['bug', 'crash', 'error', 'broken', 'fix', 'glitch', 'issue', 'problem'],
        'UX & design': ['ui', 'ux', 'design', 'interface', 'confusing', 'simple', 'easy', 'difficult'],
        'Performance': ['slow', 'fast', 'load', 'speed', 'laggy', 'performance', 'battery'],
        'Features': ['feature', 'missing', 'add', 'need', 'want', 'request', 'option', 'support'],
        'Customer support': ['support', 'help', 'response', 'team', 'service', 'contact', 'reply'],
    }

    result = {}
    for cluster_id, keywords in cluster_keywords.items():
        best_match = 'Other'
        best_score = 0
        for category, patterns in category_patterns.items():
            score = sum(1 for kw in keywords if any(p in kw for p in patterns))
            if score > best_score:
                best_score = score
                best_match = category
        result[cluster_id] = best_match

    return result


# ─── PAIN POINT EXTRACTION ────────────────────────────────────────────────────
def extract_pain_points(df: pd.DataFrame) -> list:
    """
    Find the most common words in NEGATIVE reviews only.
    These are the user pain points — the gold mine for PMF analysis.

    Returns top 20 pain point phrases, sorted by frequency.
    """
    negative_reviews = df[df['sentiment_label'] == 'negative']['clean_text']

    # Count all bigrams (2-word phrases) in negative reviews
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100, stop_words='english')

    if len(negative_reviews) < 5:
        return []

    tfidf = vectorizer.fit_transform(negative_reviews)
    word_scores = dict(zip(
        vectorizer.get_feature_names_out(),
        tfidf.sum(axis=0).A1
    ))

    # Sort by frequency and return top 20
    pain_points = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    return pain_points


# ─── FULL PIPELINE ────────────────────────────────────────────────────────────
def run_pipeline(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Runs the complete NLP pipeline on a scraped CSV.
    Call this after scraper.py has saved raw data.
    """
    print(f"\n{'=' * 50}")
    print(f"Running NLP pipeline on {input_csv}")
    print('=' * 50)

    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} records")

    # Step 1: Sentiment
    df = analyze_sentiment(df)

    # Step 2: Topic clustering
    df, cluster_keywords = cluster_topics(df, n_clusters=6)

    # Save processed data
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved processed data to {output_csv}")

    return df