# 📊 Financial News Analysis with FinBERT

A complete pipeline for analyzing **financial news sentiment** and evaluating its impact using **FinBERT**. This project processes financial headlines, performs sentiment analysis, and ranks news based on their potential impact.
---




https://github.com/user-attachments/assets/0ea048ff-63eb-4f8c-9d51-0984fcc5a914


---

## **Project Workflow**
1. **Download Data** (`ingest.py`)  
   - Fetches financial news data and saves it as a raw CSV.  
   - You can modify the script to specify your own dataset path.

2. **Preprocess Data** (`preprocess.py`)  
   - Cleans and tokenizes text using `clean_text(text)`.  
   - Removes special characters, converts text to lowercase, tokenizes, and removes stopwords.  
   - Outputs a **cleaned CSV** for sentiment analysis.

3. **Perform Sentiment Analysis** (`finbert_analysis.py`)  
   - Uses **FinBERT** to classify news sentiment as **Positive, Negative, or Neutral**.  
   - Stores results in `processed-data/finbert_results.csv`.

4. **Calculate Financial Impact** (`impact.py`)  
   - Evaluates **confidence scores** to rank news impact.  
   - Generates a ranked list in `processed-data/impact_ranked_news.csv`.

5. **End-to-End Testing** (`test.py`)  
   - Runs all scripts in sequence to verify pipeline correctness.

---

## 📂 **Project Structure**

```sh
financial-news-analysis/
├── data/
│   ├── raw/                  # Raw unprocessed news data
│   └── processed/            # Cleaned data & analysis results
├── models/                   # FinBERT pretrained model on My data all.csv
├── static/                   # CSS, JS
├── templates/               # HTML 
├── src/
│   ├── ingest.py     # Data collection script
│   ├── preprocessing.py      # Text cleaning utilities
│   ├── finbert_analysis.py # FinBERT integration
│   ├── impact.py  # News impact scoring
│   └── helpers.py            # Utility functions
├── tests/                   # Unit and integration tests
├── app.py                   # Flask web application
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration

```
---

## 🛠 **Installation & Setup**

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo/financial-news-analysis.git
```
```sh
cd financial-news-analysis
```
2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

3️⃣ Run the Application
```sh
python app.py
```

🐳 Running with Docker

build and run the container:
```sh
docker build -t financial-news .
docker run -p 5000:5000 financial-news
```
This will start the app on http://localhost:5000.

🔍 How it Works

    FinBERT Model: A fine-tuned BERT model for financial sentiment analysis.
    Impact Calculation: Uses confidence scores to determine news influence.
    Ranking System: Orders financial news by their impact score.


📜 License

This project is open-source and available under the MIT License.
