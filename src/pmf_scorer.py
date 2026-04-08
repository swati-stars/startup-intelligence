# src/pmf_scorer.py
# PURPOSE: Calculate a Product-Market Fit score (0-100) from NLP results
#
# PMF Score formula:
# A weighted combination of signals that indicate genuine market demand.
# Based on Sean Ellis's PMF framework + NLP-derived signals.
#
# Score = (sentiment_weight × 0.35) +
#         (engagement_weight × 0.25) +
#         (pain_intensity_weight × 0.25) +
#         (feature_demand_weight × 0.15)

import pandas as pd
import numpy as np


def calculate_pmf_score(df: pd.DataFrame) -> dict:
    """
    Returns a dict with:
    - pmf_score: 0-100 (higher = stronger product-market fit)
    - component scores for each dimension
    - interpretation and business recommendation
    """

    # ── Signal 1: Sentiment ratio (35% weight) ──────────────────────────────
    # High PMF products have >60% positive reviews
    # Low PMF products are dominated by complaints
    total = len(df)
    pos_pct = (df['sentiment_label'] == 'positive').sum() / total
    neg_pct = (df['sentiment_label'] == 'negative').sum() / total

    # Normalize: 0% positive = score 0, 80%+ positive = score 100
    sentiment_score = min(100, (pos_pct / 0.8) * 100)

    # ── Signal 2: Engagement quality (25% weight) ────────────────────────────
    # Long, detailed reviews = users care enough to explain
    # Short "great app" / "terrible" = low engagement
    avg_length = df['clean_text'].str.len().mean()
    # 200 chars = minimal engagement, 600+ = high engagement
    engagement_score = min(100, ((avg_length - 50) / 550) * 100)

    # ── Signal 3: Pain point intensity (25% weight) ──────────────────────────
    # Counterintuitive: strong pain points = strong market need
    # Users only complain about things they care about
    # No complaints = either no users or indifferent users
    neg_reviews = df[df['sentiment_label'] == 'negative']
    if len(neg_reviews) > 0:
        avg_pain_intensity = neg_reviews['sentiment_negative'].mean()
        # High negative intensity in complaints = real frustration = real need
        pain_score = min(100, avg_pain_intensity * 200)
    else:
        pain_score = 0

    # ── Signal 4: Feature demand (15% weight) ────────────────────────────────
    # Users requesting features = they want to stay, just need improvement
    # This is a PMF positive signal (vs users saying "I'm switching to X")
    feature_keywords = ['need', 'want', 'add', 'please', 'feature', 'wish', 'should', 'could']
    feature_mentions = df['clean_text'].str.contains(
        '|'.join(feature_keywords), case=False, na=False
    ).sum()
    feature_rate = feature_mentions / total
    feature_score = min(100, (feature_rate / 0.3) * 100)

    # ── Final weighted score ─────────────────────────────────────────────────
    pmf_score = (
            sentiment_score * 0.35 +
            engagement_score * 0.25 +
            pain_score * 0.25 +
            feature_score * 0.15
    )

    return {
        'pmf_score': round(pmf_score, 1),
        'sentiment_score': round(sentiment_score, 1),
        'engagement_score': round(engagement_score, 1),
        'pain_intensity_score': round(pain_score, 1),
        'feature_demand_score': round(feature_score, 1),
        'total_reviews': total,
        'positive_pct': round(pos_pct * 100, 1),
        'negative_pct': round(neg_pct * 100, 1),
        'interpretation': interpret_pmf(pmf_score),
        'recommendation': generate_recommendation(pmf_score, df)
    }


def interpret_pmf(score: float) -> str:
    if score >= 75:
        return "Strong PMF — users love this product and it solves a real need"
    elif score >= 55:
        return "Moderate PMF — positive signals but significant improvement areas exist"
    elif score >= 35:
        return "Weak PMF — product exists but users aren't deeply attached"
    else:
        return "No clear PMF — reconsider core value proposition"


def generate_recommendation(score: float, df: pd.DataFrame) -> str:
    """
    Generate a business recommendation based on the PMF score and data.
    This is the 'Business Insight' section that makes your project stand out.
    """
    neg_topics = df[df['sentiment_label'] == 'negative']['topic_category'].value_counts()
    top_complaint = neg_topics.index[0] if len(neg_topics) > 0 else "unclear issues"

    pos_topics = df[df['sentiment_label'] == 'positive']['topic_category'].value_counts()
    top_strength = pos_topics.index[0] if len(pos_topics) > 0 else "core functionality"

    if score >= 75:
        return (f"Product has strong PMF. Focus on scaling: invest in {top_strength} "
                f"(your biggest strength) and address {top_complaint} to reduce churn further.")
    elif score >= 55:
        return (f"PMF emerging but not locked in. Primary blocker: {top_complaint}. "
                f"Fix this before scaling marketing spend. "
                f"Users value {top_strength} — double down here.")
    else:
        return (f"PMF not yet achieved. Major blocker: {top_complaint}. "
                f"Recommend founder interviews to validate whether {top_complaint} "
                f"can be resolved within current product scope.")