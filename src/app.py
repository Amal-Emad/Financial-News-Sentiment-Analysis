from flask import Flask, render_template, request, jsonify
import pandas as pd
import torch
import scipy.special
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Load FinBERT model
model_path = r"C:\Users\user\financial_news_analysis\data\models\finbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

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
    return confidence * (105 if sentiment == "positive" else 100 if sentiment == "negative" else 30)

def process_csv(file_path):
    df = pd.read_csv(file_path)
    if "headline" not in df.columns:
        return None  
    results = []
    for headline in df["headline"].dropna():
        sentiment, confidence = analyze_sentiment(headline)
        impact_score = calculate_impact(sentiment, confidence)
        results.append([headline, sentiment, confidence, impact_score])
    return sorted(results, key=lambda x: x[3], reverse=True)

def process_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    
    headlines = [line.strip() for line in text.split("\n") if len(line.strip()) > 10]
    results = []
    for headline in headlines[:1000]:  # -----Limit processing to first 1000 valid headlines----
        sentiment, confidence = analyze_sentiment(headline)
        impact_score = calculate_impact(sentiment, confidence)
        results.append([headline, sentiment, confidence, impact_score])
    return sorted(results, key=lambda x: x[3], reverse=True)

@app.route("/", methods=["GET", "POST"])
def index():
    news_data = []
    error_message = None
    
    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            
            if file.filename.endswith(".csv"):
                news_data = process_csv(file_path)
            elif file.filename.endswith(".pdf"):
                news_data = process_pdf(file_path)
            else:
                error_message = "Unsupported file format. Please upload a CSV or PDF file."
                
            os.remove(file_path) 
        else:
            news_headline = request.form.get("news_headline")
            if news_headline:
                sentiment, confidence = analyze_sentiment(news_headline)
                impact_score = calculate_impact(sentiment, confidence)
                news_data.append([news_headline, sentiment, confidence, impact_score])

    return render_template("index.html", news_data=news_data, error_message=error_message)

if __name__ == "__main__":
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    app.run(debug=True)
