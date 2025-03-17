import pandas as pd

''' 
#========================================================
# Step 1: Load Sentiment Analysis Results
#========================================================
# - Reads the CSV file containing sentiment predictions.
# - The file should have columns: ["Predicted Sentiment", "Confidence"].
#========================================================
'''
results_file = r'C:\Users\user\financial_news_analysis\src\data\finbert_results.csv'
df = pd.read_csv(results_file)


''' 
#========================================================
# Step 2: Define Impact Scoring Function
#========================================================
# - Assigns an impact score based on sentiment and confidence.
# - Higher weight for positive sentiment, slightly lower for negative.
# - Neutral sentiment has a significantly lower weight.
# - Normalizes the score within a 0-100 range.
#========================================================
'''
def calculate_impact(sentiment, confidence):
    if sentiment == "positive":
        weight = 1.0  # Highest weight for positive sentiment
    elif sentiment == "negative":
        weight = 0.97  # Slightly lower weight for negative sentiment
    else:
        weight = 0.2  # Significantly lower weight for neutral sentiment
    
    return confidence * weight * 100  # Normalize score to a 0-100 range


''' 
#========================================================
# Step 3: Apply Impact Score Calculation
#========================================================
# - Applies the `calculate_impact` function to each row.
# - Adds a new column "Impact Score" to store the computed values.
#========================================================
'''
df["Impact Score"] = df.apply(lambda row: calculate_impact(row["Predicted Sentiment"], row["Confidence"]), axis=1)


''' 
#========================================================
# Step 4: Rank News Articles by Impact Score
#========================================================
# - Sorts the news from highest to lowest impact.
# - Ensures the most influential articles appear at the top.
#========================================================
'''
df = df.sort_values(by="Impact Score", ascending=False)


''' 
#========================================================
# Step 5: Save Ranked Results
#========================================================
# - Exports the ranked news to a new CSV file.
# - The output file is saved in the project directory.
#========================================================
'''
output_file = r'C:\Users\user\financial_news_analysis\src\data\impact_ranked_news.csv'
df.to_csv(output_file, index=False)

print(f"\nRanked impact news saved successfully to: {output_file}")
