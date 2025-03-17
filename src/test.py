import pandas as pd
import torch
import scipy.special
from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
This script analyzes the sentiment of trending financial news headlines using a FinBERT model.  
It follows these steps:

1. load the FinBERT model and tokenizer** from a local directory.
2.Define a list of financial news headlines** from 2025.
3.Analyze sentiment** for each headline using the model, returning:
   - Sentiment (Positive, Negative, Neutral)
   - Confidence score (probability of the assigned sentiment)
4.Calculate an impact score** based on sentiment confidence:
   - Positive sentiment: confidence × 105
   - Negative sentiment: confidence × 100
   - Neutral sentiment: confidence × 30
5.Store results in a Pandas DataFrame** with sentiment, confidence, and impact scores.
6.Rank news by impact score** in descending order.
7.Print and save the results** to a CSV file.

The script processes headlines individually but could be optimized to batch-process them for efficiency.
"""

# Load FinBERT model and tokenizer
model_path = r"C:\Users\user\financial_news_analysis\src\data\models\finbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Trending financial news in 2025
news_headlines = [
    "Tesla Unveils Fully Autonomous Taxi Fleet Stock Soars 20%",
    "Bitcoin Hits $150,000 Amid Growing Institutional Adoption",
    "US Fed Cuts Interest Rates to 2% to Combat Recession Fears",
    "Apple Announces AI-Powered iPhone, Revolutionizing Mobile Tech",
    "Global Oil Prices Surge After Supply Chain Disruptions",
    "Ethereum Surpasses $10,000 as DeFi Boom Continues",
    "Stock Market Experiences Largest Single-Day Crash in 10 Years",
    "Saudi Arabia Launches World's Largest Green Hydrogen Project",
    "Meta's AI Glasses Replace Smartphones, Changing Tech Landscape",
    "Gold Prices Skyrocket as Investors Seek Safe Haven Assets",
    "Google s Quantum AI Outperforms Supercomputers for Financial Trading",
    "Elon Musk's Neuralink Begins Human Trials, Stock Value Rises",
    "China Overtakes US as World's Largest Economy",
    "Amazon Enters Healthcare Industry, Disrupting Traditional Medicine",
    "Dubai Introduces Crypto-Based Real Estate Transactions"
]


def analyze_sentiment(text):
    with torch.no_grad():
        input_sequence = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logits = model(**input_sequence).logits  
        scores = {
            k: v for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze())
            )
        }
        sentiment = max(scores, key=scores.get)
        confidence = max(scores.values())
    return sentiment, confidence


def calculate_impact(sentiment, confidence):
    if sentiment == "positive":
        return confidence * 105  
    elif sentiment == "negative":
        return confidence * 100  
    else:  # Neutral
        return confidence * 30  


news_data = []
for headline in news_headlines:
    sentiment, confidence = analyze_sentiment(headline)
    impact_score = calculate_impact(sentiment, confidence)
    news_data.append([headline, sentiment, confidence, impact_score])


df = pd.DataFrame(news_data, columns=["Headline", "Sentiment", "Confidence", "Impact Score"])


df = df.sort_values(by="Impact Score", ascending=False)


print(df.to_string(index=False))

output_file = r'C:\Users\user\financial_news_analysis\src\ranked_news_2025.csv'
df.to_csv(output_file, index=False)
print(f"\nRanked news saved to: {output_file}")
