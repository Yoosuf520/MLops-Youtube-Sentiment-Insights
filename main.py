import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd

import os
import pickle
import sys
import nltk
import mlflow
import base64
import io
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# For visual chart generation
import matplotlib
matplotlib.use('Agg')  # Prevents GUI rendering issues inside the Docker container
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure NLTK data path points to where it is pre-downloaded in the Docker layer
nltk.data.path.append('/usr/local/share/nltk_data')
from nltk.corpus import stopwords

# Initialize FastAPI App
app = FastAPI(
    title="YouTube Sentiment Insights API",
    description="Production API serving LightGBM predictions and visual analytics for Chrome Extension",
    version="1.0.0"
)

# Raw comment object schema
class CommentItem(BaseModel):
    text: str
    published_at: str  # ISO timestamp from YouTube API (e.g., "2026-07-02T01:19:00Z")

# Updated input payload schema to accept text + metadata
class CommentPayload(BaseModel):
    comments: List[CommentItem]

# ==========================================
# 1. ARTIFACT RESOLUTION ENGINE
# ==========================================
def load_model_and_vectorizer(model_uri: str):
    print(f"Connecting to MLflow Tracking Server to fetch: {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)
    
    try:
        local_artifacts_dir = model.metadata.get_model_info()._download_dir
    except Exception:
        from mlflow.tracking.artifact_utils import _download_artifact_from_uri
        local_artifacts_dir = _download_artifact_from_uri(model_uri)
        
    vectorizer_path = os.path.join(local_artifacts_dir, "tfidf_vectorizer.pkl")
    print(f"Loading text vectorizer binary from: {vectorizer_path}")
    
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Could not find tfidf_vectorizer.pkl at evaluated path: {vectorizer_path}")
        
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
        
    return model, vectorizer

# ==========================================
# 2. MODEL RUNTIME INITIALIZATION
# ==========================================
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(tracking_uri)
MODEL_URI = "models:/youtube_chrome_plugin_model/1"

try:
    model, vectorizer = load_model_and_vectorizer(MODEL_URI)
    print("Application successfully initialized. Model weights and features loaded perfectly.")
except Exception as e:
    print(f"CRITICAL FAULT: Server lifecycle terminated during initialization pipeline setup. Error: {e}")
    sys.exit(1)

# ==========================================
# 3. ANALYTICS & VISUALIZATION HELPERS
# ==========================================
def generate_sentiment_chart(pos_count: int, neg_count: int, neu_count: int) -> str:
    """Generates a donut chart and encodes it to a Base64 string for the Chrome extension UI."""
    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [pos_count, neg_count, neu_count]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    # Filter out categories with zero counts to keep the chart clean
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
    """Generates a text word cloud image string from high-frequency processed comment words."""
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
    ax.imshow(wordcloud, interpolation=' Harlow ')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def parse_timestamp(ts_str: str) -> int:
    """Parses standard ISO strings into hour integers to model time-of-day workflow patterns."""
    try:
        # Matches "2026-07-02T01:19:00Z"
        clean_ts = re.sub(r'\.\d+Z$', 'Z', ts_str)
        dt = datetime.strptime(clean_ts, "%Y-%m-%dT%H:%M:%SZ")
        return dt.hour
    except Exception:
        return 12  # Fallback to noon on structural parse error

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
        
        # 1. Dispatch Model Inference
        transformed_features = vectorizer.transform(raw_texts)
        predictions = model.predict(transformed_features)
        
        # 2. Track metrics and process timeline distributions
        pos, neg, neu = 0, 0, 0
        hourly_workload = {i: 0 for i in range(24)}  # Tracks comments grouped by hour of day
        combined_text = ""
        
        results = []
        for item, pred in zip(payload.comments, predictions):
            sentiment_val = int(pred)
            
            # Count distribution labels (Assuming 1=Positive, 0=Negative, 2=Neutral. Adjust if different!)
            if sentiment_val == 1:
                pos += 1
            elif sentiment_val == 0:
                neg += 1
            else:
                neu += 1
                
            # Process timestamps for hourly activity mapping
            hour = parse_timestamp(item.published_at)
            hourly_workload[hour] += 1
            
            # Append text corpus for word cloud generation
            combined_text += f" {item.text}"
            
            results.append({
                "text": item.text,
                "sentiment": sentiment_val,
                "hour": hour
            })
            
        # 3. Generate Visual Assets
        chart_base64 = generate_sentiment_chart(pos, neg, neu)
        wordcloud_base64 = generate_wordcloud(combined_text)
        
        # 4. Return complete packaged payload for extension UI consumption
        return {
            "success": True,
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
            "raw_predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {str(e)}")



if __name__ == "__main__":
    
    import uvicorn
    # This runs the FastAPI instance 'app' on port 8000 when executing 'python main.py'
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)