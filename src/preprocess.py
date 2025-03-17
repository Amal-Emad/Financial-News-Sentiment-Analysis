import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords


nltk.download('all')
nltk.download('stopwords')
nltk.download('punkt')


stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Cleans and preprocesses financial text data.
    
    Steps:
    - Removes special characters, keeping only letters and spaces.
    - Converts text to lowercase and trims whitespace.
    - Tokenizes text into individual words.
    - Removes common English stopwords.
    
    Args:
        text (str): The raw text input.
    
    Returns:
        str: The cleaned and tokenized text as a single string.
    """
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
    text = text.lower().strip()  #Normalize case and remove leading/trailing spaces
    words = nltk.word_tokenize(text) # Tokenize text into words
    return ' '.join(word for word in words if word not in stop_words)  #Remove stopwords

def preprocess_data(input_path, output_path):
    """
    Reads, processes, and saves financial news data.
    
    Steps:
    - Loads the dataset from CSV.
    - Converts text column to string type.
    - Applies text cleaning function to preprocess content.
    - Saves the processed dataset to a new CSV file.
    
    Args:
        input_path (str): File path to the raw dataset.
        output_path (str): File path to save the cleaned dataset.
    
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    df = pd.read_csv(input_path, encoding='latin-1', header=None, names=['sentiment', 'text'])
    df['clean_text'] = df['text'].astype(str).apply(clean_text)  # Ensure text column is a string before processing

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

   
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    
    return df

if __name__ == "__main__":
    
    input_file = r"C:\Users\user\financial_news_analysis\data\all-data.csv"
    output_file = r"C:\Users\user\financial_news_analysis\data\processed_data.csv"
    
    
    processed_data = preprocess_data(input_file, output_file)
    
   
    print("\nSample cleaned data:")
    print(processed_data[['text', 'clean_text']].head())
