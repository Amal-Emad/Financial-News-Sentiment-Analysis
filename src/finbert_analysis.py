''' 
Step 1: Import necessary libraries
- pandas: for data handling
- scipy: for numerical operations
- seaborn & matplotlib: for visualization
- sklearn: for evaluation metrics
- torch: for deep learning (PyTorch)
- transformers: for loading FinBERT (pre-trained NLP model)
'''
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

''' 
Step 2: Load the processed financial data
- Ensure correct column names while loading the dataset
- Use 'unicode_escape' encoding to avoid special character issues
'''
file_path = r'C:\Users\user\financial_news_analysis\src\data\processed_data.csv'

# Load dataset with correct column names
data = pd.read_csv(file_path, encoding='unicode_escape')
print("Dataset Loaded Successfully!")

''' 
Step 3: Inspect the dataset to ensure column names are correct
'''
print("First 5 rows of the dataset:")
print(data.head())

print("\nColumn Names:")
print(data.columns)

''' 
Step 4: Fix column names issue
- Ensure correct column names (case-sensitive)
- Rename columns to standardized names if needed
'''
# Rename columns if necessary
data.rename(columns={'sentiment': 'Sentiment', 'clean_text': 'Text'}, inplace=True)

# Extract input texts and target labels
X = data['Text'].tolist()  # Financial news text
y = data['Sentiment'].tolist()  # Corresponding sentiment labels

''' 
Step 5: Load the FinBERT model and tokenizer from Hugging Face
- AutoTokenizer: Prepares text for input into FinBERT
- AutoModelForSequenceClassification: Loads FinBERT model for sentiment analysis
'''
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

print("\nFinBERT Model and Tokenizer Loaded Successfully!")

''' 
Step 6: Perform Sentiment Analysis using FinBERT
- Tokenize each input text
- Run the model to obtain sentiment predictions
'''
preds = []
preds_proba = []

# Tokenization settings
tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}

print("\nStarting Sentiment Analysis...")

print("\nStarting Sentiment Analysis...")

for i, text in enumerate(X):
    text = str(text).strip()  # Ensure text is a clean string

    if not text:  # Skip empty strings
        continue  # Do NOT add anything to preds or preds_proba

    with torch.no_grad():  
        input_sequence = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        logits = model(**input_sequence).logits  

        scores = {
            k: v for k, v in zip(
                model.config.id2label.values(),
                scipy.special.softmax(logits.numpy().squeeze())
            )
        }

        sentimentFinbert = max(scores, key=scores.get)
        probabilityFinbert = max(scores.values())

        preds.append(sentimentFinbert)
        preds_proba.append(probabilityFinbert)

    if i % 10 == 0:
        print(f"Processed {i+1}/{len(X)} texts...")

print("\nSentiment Analysis Completed!")


''' 
Step 7: Evaluate the model’s performance
- Calculate Accuracy
- Generate a Classification Report
'''
print("\nModel Performance:")
accuracy = accuracy_score(y, preds)
print(f'Accuracy Score: {accuracy:.4f}')
print("\nClassification Report:")
print(classification_report(y, preds))

''' 
Step 8: Save results for further analysis
- Create a new DataFrame with original text, predicted sentiment, and confidence scores
- Save results to a CSV file
'''
results_df = pd.DataFrame({'Text': X, 'Actual Sentiment': y, 'Predicted Sentiment': preds, 'Confidence': preds_proba})
results_df.to_csv(r'C:\Users\user\financial_news_analysis\src\finbert_results.csv', index=False)

print("\nResults Saved Successfully to 'finbert_results.csv'!")
# Save the FinBERT model and tokenizer
model.save_pretrained("finbert_model")  # Saves model weights
tokenizer.save_pretrained("finbert_model")  # Saves tokenizer
print("\nModel saved successfully in 'finbert_model/'!")
