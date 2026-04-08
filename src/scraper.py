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
    df = pd.DataFrame(result)
    print(f"Available columns: {df.columns.tolist()}")
    keep_cols = ['content', 'score', 'at']
    for optional in ['thumbsUpCount', 'thumbsUp']:
        if optional in df.columns:
            keep_cols.append(optional)
            break
    df = df[keep_cols]
    df['source'] = 'play_store'
    df['app_name'] = app_name
    df['text'] = df['content']
    print(f"  Collected {len(df)} reviews")
    return df

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
