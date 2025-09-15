import snscrape.modules.twitter as sntwitter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# === SETTINGS ===
query = "S&P 500"
since_date = "2024-12-01"
until_date = datetime.today().strftime('%Y-%m-%d')
max_tweets = 200  # You can increase this for more data

# === SCRAPE TWEETS ===
tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} since:{since_date} until:{until_date}').get_items()):
    if i >= max_tweets:
        break
    tweets.append({
        "date": tweet.date.date(),
        "content": tweet.content
    })

# === SENTIMENT ANALYSIS ===
analyzer = SentimentIntensityAnalyzer()
for tweet in tweets:
    tweet["sentiment"] = analyzer.polarity_scores(tweet["content"])["compound"]

df = pd.DataFrame(tweets)
sentiment_df = df.groupby("date")["sentiment"].mean().reset_index()

# === S&P 500 RETURNS ===
start_date = sentiment_df["date"].min() - timedelta(days=1)
sp500 = yf.download("^GSPC", start=start_date, end=until_date)
sp500 = sp500["Adj Close"].pct_change().reset_index()
sp500.columns = ["date", "returns"]
sp500["date"] = sp500["date"].dt.date

# === MERGE & PLOT ===
merged = pd.merge(sentiment_df, sp500, on="date", how="inner")

plt.figure(figsize=(12,6))
plt.plot(merged["date"], merged["sentiment"], label="Twitter Sentiment", marker='o')
plt.plot(merged["date"], merged["returns"], label="S&P 500 Returns", marker='x')
plt.axhline(0, color="gray", linestyle="--")
plt.title("Twitter Sentiment vs S&P 500 Returns")
plt.xlabel("Date")
plt.ylabel("Score / Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()