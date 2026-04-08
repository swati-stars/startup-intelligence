# Startup Intelligence Suite 🎯

> **Two NLP-powered dashboards that turn messy app reviews into startup insights**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-VADER-green?style=flat)](https://github.com/cjhutto/vaderSentiment)
[![ML](https://img.shields.io/badge/ML-scikit--learn-orange?style=flat)](https://scikit-learn.org)

---

## Live Demos

| Dashboard | Link |
|-----------|------|
| 🎯 PMF Analyzer | [Launch →](YOUR_PMF_STREAMLIT_URL) |
| 📊 Feedback Intelligence | [Launch →](YOUR_FEEDBACK_STREAMLIT_URL) |

---

## What This Does

Most startups fail because they build something nobody wants. This suite helps founders and analysts answer two critical questions — **before** spending money on growth:

**Dashboard 1 — PMF Analyzer**
Analyzes any Google Play app's reviews using NLP and produces a 0–100 Product-Market Fit score. Tells you whether a product has genuine market demand, and what to fix if it doesn't.

**Dashboard 2 — Feedback Intelligence Dashboard**
Converts thousands of messy reviews into structured complaint categories with percentages. Shows exactly what % of users are complaining about pricing, bugs, UX, performance, and more.

---

## PMF Score Formula

The PMF score is an original weighted formula combining 4 signals:

```
PMF Score =
  (sentiment_score    × 0.35) +
  (engagement_score   × 0.25) +
  (pain_intensity     × 0.25) +
  (feature_demand     × 0.15)
```

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Sentiment ratio | 35% | % of positive vs negative reviews |
| Engagement quality | 25% | How detailed reviews are (depth of care) |
| Pain intensity | 25% | How intense negative reviews are (attached users complain loudly) |
| Feature demand | 15% | How many users request features (want to stay) |

**Score interpretation:**
- 75–100 → Strong PMF. Scale now.
- 55–74 → Moderate PMF. Fix top complaint first.
- 35–54 → Weak PMF. Talk to users before spending.
- 0–34  → No PMF. Reconsider core value proposition.

---

## Example Output

Analyzing **Notion** (500 reviews):

```
PMF Score:        72 / 100  (Strong PMF)
Positive reviews: 68%
Top complaint:    Feature gaps (29% of negative reviews)

Recommendation: Product has strong PMF. Focus on scaling.
Address feature gaps to reduce churn further.
Users value core editor experience — double down here.
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `google-play-scraper` | Scrape app reviews without API key |
| `praw` | Reddit API for startup discussions |
| `VADER` | Sentiment analysis (optimized for reviews) |
| `scikit-learn` | TF-IDF vectorization + KMeans topic clustering |
| `pandas` | Data manipulation and CSV handling |
| `Streamlit` | Interactive web dashboard |
| `Plotly` | Interactive charts and gauges |
| `python-dotenv` | Secure API key management |

---

## Project Structure

```
startup-intelligence/
├── src/
│   ├── scraper.py          # Google Play + Reddit data collection
│   ├── nlp_pipeline.py     # Sentiment analysis + topic clustering
│   └── pmf_scorer.py       # Original PMF scoring formula
├── dashboards/
│   ├── pmf_dashboard.py    # Dashboard 1: PMF Analyzer
│   └── feedback_dashboard.py  # Dashboard 2: Feedback Intelligence
├── data/                   # Scraped + processed CSVs (gitignored)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/startup-intelligence
cd startup-intelligence
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your Reddit API credentials (optional)
```

**5. Scrape data and run NLP pipeline**
```bash
python3 src/scraper.py
python3 run.py
```

**6. Launch dashboards**
```bash
# Dashboard 1
streamlit run dashboards/pmf_dashboard.py

# Dashboard 2 (new terminal)
streamlit run dashboards/feedback_dashboard.py --server.port 8502
```

---

## Analyze Any App

Find any app's ID from its Google Play URL:
```
https://play.google.com/store/apps/details?id=THIS_IS_THE_ID
```

Examples:
| App | ID |
|-----|----|
| Notion | `notion.id` |
| Spotify | `com.spotify.music` |
| Duolingo | `com.duolingo` |
| Trello | `com.trello` |
| Swiggy | `in.swiggy.android` |

---

## Business Insight

This project demonstrates that NLP-driven PMF analysis is a real market need:

- Companies like **Medallia**, **Qualtrics**, and **Appbot** charge $500–$5,000/month for enterprise versions of this capability
- 42% of startups fail due to lack of market need — PMF analysis directly addresses this
- The global NLP market is projected to reach **$9.4B by 2026**

---

## Skills Demonstrated

`Web Scraping` · `NLP` · `Sentiment Analysis` · `ML Clustering` · `Data Visualization` · `Dashboard Development` · `API Integration` · `Business Analytics` · `Python` · `Streamlit`

---

## Author

**Your Name**
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## License

MIT License — free to use and modify.
