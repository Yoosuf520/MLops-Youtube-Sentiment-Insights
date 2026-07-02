import os
import io
import re
import sys
import base64
import pickle
from datetime import datetime
from typing import List

import nltk
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.data.path.append('/usr/local/share/nltk_data')
from nltk.corpus import stopwords

app = FastAPI(
    title="YouTube Sentiment Insights API",
    description="Production API serving LightGBM predictions and visual analytics for Chrome Extension",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentItem(BaseModel):
    text: str
    published_at: str

class CommentPayload(BaseModel):
    comments: List[CommentItem]

# ==========================================
# 1. ARTIFACT RESOLUTION ENGINE
# ==========================================
def load_model_and_vectorizer(model_uri: str):
    print(f"Connecting to MLflow Tracking Server to fetch: {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)

    # Fix: Load vectorizer from local app directory directly
    vectorizer_path = "/app/tfidf_vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        # Fallback to current directory
        vectorizer_path = "./tfidf_vectorizer.pkl"
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"tfidf_vectorizer.pkl not found. Make sure it is copied into the Docker image."
        )

    print(f"Loading vectorizer from: {vectorizer_path}")
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)

    return model, vectorizer

# ==========================================
# 2. MODEL RUNTIME INITIALIZATION
# ==========================================
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://3.110.185.110:5000")
mlflow.set_tracking_uri(tracking_uri)
MODEL_URI = "models:/youtube_chrome_plugin_model/1"

try:
    model, vectorizer = load_model_and_vectorizer(MODEL_URI)
    print("Application successfully initialized.")
except Exception as e:
    print(f"CRITICAL FAULT: {e}")
    sys.exit(1)

# ==========================================
# 3. ANALYTICS & VISUALIZATION HELPERS
# ==========================================
def generate_sentiment_chart(pos_count: int, neg_count: int, neu_count: int) -> str:
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos_count, neg_count, neu_count]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']

    filtered_data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if not filtered_data:
        return ""

    lbls, szs, cls = zip(*filtered_data)
    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        szs, labels=lbls, autopct='%1.1f%%', startangle=90,
        colors=cls, wedgeprops=dict(width=0.4, edgecolor='w')
    )
    plt.setp(autotexts, size=10, weight="bold", color="white")
    plt.setp(texts, size=10)
    ax.axis('equal')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generate_wordcloud(text_corpus: str) -> str:
    if not text_corpus.strip():
        return ""
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = None

    wordcloud = WordCloud(
        width=400, height=200, background_color=None, mode="RGBA",
        max_words=50, stopwords=stop_words, colormap='viridis'
    ).generate(text_corpus)

    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def parse_timestamp(ts_str: str) -> int:
    try:
        clean_ts = re.sub(r'\.\d+Z$', 'Z', ts_str)
        dt = datetime.strptime(clean_ts, "%Y-%m-%dT%H:%M:%SZ")
        return dt.hour
    except Exception:
        return 12

# ==========================================
# 4. FASTAPI API ROUTING
# ==========================================
@app.get("/")
def health_check():
    return {"status": "healthy", "model_registry_target": MODEL_URI}

@app.post("/predict")
def predict_sentiment(payload: CommentPayload):
    if not payload.comments:
        raise HTTPException(status_code=400, detail="Comment payload array cannot be empty.")

    try:
        raw_texts = [item.text for item in payload.comments]

        transformed_features = vectorizer.transform(raw_texts)
        predictions = model.predict(transformed_features)

        # Fix: correct sentiment mapping
        # 1 = positive, 0 = neutral, -1 = negative
        pos, neg, neu = 0, 0, 0
        hourly_workload = {i: 0 for i in range(24)}
        combined_text = ""

        results = []
        for item, pred in zip(payload.comments, predictions):
            sentiment_val = int(pred)

            if sentiment_val == 1:
                pos += 1
            elif sentiment_val == -1:
                neg += 1
            else:
                neu += 1

            hour = parse_timestamp(item.published_at)
            hourly_workload[hour] += 1
            combined_text += f" {item.text}"

            results.append({
                "comment": item.text,
                "sentiment": str(sentiment_val),
                "timestamp": item.published_at,
                "hour": hour
            })

        chart_base64 = generate_sentiment_chart(pos, neg, neu)
        wordcloud_base64 = generate_wordcloud(combined_text)

        return {
            "success": True,
            "predictions": results,
            "metrics": {
                "total_comments": len(payload.comments),
                "positive": pos,
                "negative": neg,
                "neutral": neu
            },
            "visualizations": {
                "sentiment_donut_chart": chart_base64,
                "wordcloud_chart": wordcloud_base64
            },
            "time_distribution_workload": hourly_workload,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)