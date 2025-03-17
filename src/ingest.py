import pandas as pd

def load_data(file_path):
    
    df = pd.read_csv(
        file_path, 
        header=None,           
        names=['Sentiment', 'News Headline'], 
        encoding='latin-1'    
    )
    return df

if __name__ == "__main__":
    df = load_data(r"C:\Users\user\Downloads\all-data.csv")
    print(df.head(10))