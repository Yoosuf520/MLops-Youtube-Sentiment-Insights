import os
import io
import re
import sys
import base64
import pickle
from datetime import datetime
from typing import List, Dict, Any

import nltk
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# For visual chart generation configurations
import matplotlib
matplotlib.use('Agg')  # Prevents GUI rendering issues inside the Docker container
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Ensure NLTK data path points to where it is pre-downloaded in the Docker layer
nltk.data.path.append('/usr/local/share/nltk_data')
from nltk.corpus import stopwords

# Initialize FastAPI App instance
app = FastAPI(
    title="YouTube Sentiment Insights API",
    description="Production API serving LightGBM predictions and visual analytics for Chrome Extension",
    version="1.0.0"
)

# Enable CORS Middleware to accept connection streams from the Chrome Extension Origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Raw comment inbound structural schema
class CommentItem(BaseModel):
    text: str
    published_at: str  # ISO timestamp string format

# Payload configuration input schema
class CommentPayload(BaseModel):
    comments: List[CommentItem]

# ==========================================
# 1. ARTIFACT RESOLUTION ENGINE
# ==========================================
def load_model_and_vectorizer(model_uri: str):
    """
    Loads the PyFunc model from MLflow tracking server and dynamically
    scans the downloaded artifact tree to find and load the vectorizer.
    """
    print(f"Connecting to MLflow Tracking Server to fetch: {model_uri} ...")
    model = mlflow.pyfunc.load_model(model_uri)
    
    try:
        local_artifacts_dir = model.metadata.get_model_info()._download_dir
    except Exception:
        from mlflow.artifacts import download_artifacts
        local_artifacts_dir = download_artifacts(artifact_uri=model_uri)
        
    print(f"Artifact directory downloaded to: {local_artifacts_dir}")
    
    # Dynamically scan the directory tree to look for the file wrapper
    vectorizer_path = None
    for root, dirs, files in os.walk(local_artifacts_dir):
        if "tfidf_vectorizer.pkl" in files:
            vectorizer_path = os.path.join(root, "tfidf_vectorizer.pkl")
            break
            
    if not vectorizer_path:
        all_files = []
        for r, d, f in os.walk(local_artifacts_dir):
            for file in f:
                all_files.append(os.path.relpath(os.path.join(r, file), local_artifacts_dir))
        raise FileNotFoundError(
            f"Could not find tfidf_vectorizer.pkl anywhere inside {local_artifacts_dir}. "
            f"Available files in artifact structure: {all_files}"
        )
        
    print(f"Success! Found and loading text vectorizer from resolved path: {vectorizer_path}")
    
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
    # ─── FIXED: Changed interpolation scheme from 'Harlow' to standard 'bilinear' ───
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def parse_timestamp(ts_str: str) -> int:
    """Parses standard ISO strings into hour integers to model time-of-day workflow patterns."""
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
        
        # 1. Dispatch Model Inference Matrix
        transformed_features = vectorizer.transform(raw_texts)
        predictions = model.predict(transformed_features)
        
        # 2. Track metrics and process timeline distributions
        pos, neg, neu = 0, 0, 0
        hourly_workload = {i: 0 for i in range(24)}  
        combined_text = ""
        
        results = []
        for item, pred in zip(payload.comments, predictions):
            sentiment_val = int(pred)
            
            if sentiment_val == 1:
                pos += 1
            elif sentiment_val == 0:
                neg += 1
            else:
                neu += 1
                
            hour = parse_timestamp(item.published_at)
            hourly_workload[hour] += 1
            combined_text += f" {item.text}"
            
            results.append({
                "text": item.text,
                "sentiment": sentiment_val,
                "hour": hour
            })
            
        # 3. Generate Visual Base64 Assets
        chart_base64 = generate_sentiment_chart(pos, neg, neu)
        wordcloud_base64 = generate_wordcloud(combined_text)
        
        # 4. Return packaged analytics payload for UI consumption
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

# ==========================================
# 5. LOCAL RUNTIME ENTRYPOINT
# ==========================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)