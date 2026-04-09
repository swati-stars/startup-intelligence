import pandas as pd
import praw
from google_play_scraper import reviews, Sort
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def scrape_play_store(app_id, app_name, count=500):
    print(f"Scraping {count} reviews for {app_name}...")
    result, _ = reviews(app_id, lang='en', country='us', sort=Sort.MOST_RELEVANT, count=count)
    
    if not result:
        print("  No results returned")
        return pd.DataFrame(columns=['text', 'score', 'at', 'source', 'app_name'])
    
    df = pd.DataFrame(result)
    print(f"  Actual columns returned: {df.columns.tolist()}")
    
    # Find the review text column — could be named differently
    text_col = None
    for candidate in ['content', 'text', 'review', 'body', 'reviewText']:
        if candidate in df.columns:
            text_col = candidate
            break
    
    # Find the rating column
    score_col = None
    for candidate in ['score', 'rating', 'stars', 'reviewRating']:
        if candidate in df.columns:
            score_col = candidate
            break

    # Find the date column
    date_col = None
    for candidate in ['at', 'date', 'reviewCreatedVersion', 'reviewDate']:
        if candidate in df.columns:
            date_col = candidate
            break

    if not text_col:
        print(f"  Could not find text column in: {df.columns.tolist()}")
        # Use first string column as fallback
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break

    print(f"  Using: text={text_col}, score={score_col}, date={date_col}")

    result_df = pd.DataFrame()
    result_df['text'] = df[text_col] if text_col else ''
    result_df['score'] = df[score_col] if score_col else 3
    result_df['at'] = df[date_col] if date_col else pd.Timestamp.now()
    result_df['source'] = 'play_store'
    result_df['app_name'] = app_name

    print(f"  Collected {len(result_df)} reviews")
    return result_df

def scrape_reddit(query, subreddits, limit=200):
    reddit = praw.Reddit(client_id=os.getenv('REDDIT_CLIENT_ID'), client_secret=os.getenv('REDDIT_CLIENT_SECRET'), user_agent='startup_pmf_analyzer/1.0')
    records = []
    for sub_name in subreddits:
        print(f"  Searching r/{sub_name} for '{query}'...")
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.search(query, limit=limit // len(subreddits)):
            records.append({'text': post.title + ' ' + post.selftext, 'score': post.score, 'at': datetime.fromtimestamp(post.created_utc), 'source': 'reddit', 'subreddit': sub_name, 'app_name': query})
    df = pd.DataFrame(records)
    print(f"  Collected {len(df)} Reddit posts")
    return df

def collect_data(app_id, app_name, reddit_query=None):
    dfs = []
    play_df = scrape_play_store(app_id, app_name, count=500)
    dfs.append(play_df)
    reddit_id = os.getenv('REDDIT_CLIENT_ID')
    if reddit_id and reddit_id != 'your_client_id_here':
        print("Reddit credentials found, scraping Reddit...")
        reddit_df = scrape_reddit(reddit_query, subreddits=['startups', 'SaaS', 'entrepreneur', 'androidapps'], limit=200)
        dfs.append(reddit_df)
    else:
        print("No Reddit credentials yet — using Play Store data only")
    combined = pd.concat(dfs, ignore_index=True)
    combined.dropna(subset=['text'], inplace=True)
    combined['text'] = combined['text'].astype(str).str.strip()
    os.makedirs('data', exist_ok=True)
    filename = f"data/{app_name.lower().replace(' ', '_')}_raw.csv"
    combined.to_csv(filename, index=False)
    print(f"\n✓ Saved {len(combined)} records to {filename}")
    return combined

if __name__ == "__main__":
    collect_data(app_id="notion.id", app_name="Notion", reddit_query="Notion app productivity")
