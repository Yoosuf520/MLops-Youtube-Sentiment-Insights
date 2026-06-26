import matplotlib
matplotlib.use('Agg')

import os  # 🌟 Added for dynamic environment tracking variables
import io
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from wordcloud import WordCloud
import mlflow
import mlflow.pyfunc
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient

app = FastAPI()

# Enable CORS for all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Models ──────────────────────────────────────────────────────────

class Comment(BaseModel):
    text: str
    timestamp: Optional[str] = None

class PredictRequest(BaseModel):
    comments: List[str]

class PredictWithTimestampsRequest(BaseModel):
    comments: List[Comment]

class GenerateChartRequest(BaseModel):
    sentiment_counts: dict

class GenerateWordCloudRequest(BaseModel):
    comments: List[str]

class SentimentDataItem(BaseModel):
    timestamp: str
    sentiment: str

class GenerateTrendRequest(BaseModel):
    sentiment_data: List[SentimentDataItem]

# ─── Preprocessing ────────────────────────────────────────────────────────────

def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])
        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# ─── Load Model and Vectorizer ────────────────────────────────────────────────

def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str):
    """Load the model dynamically using the environment variables."""
    # 🌟 CRITICAL FIX: Read tracking URI from container context safely
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://13.126.127.213:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

# Initialize model and vectorizer on startup
model, vectorizer = load_model_and_vectorizer(
    "yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl"
)

# ─── Helper function — transform comments to DataFrame ───────────────────────

def transform_to_dataframe(comments: List[str]) -> pd.DataFrame:
    """
    Transforms preprocessed comments to a DataFrame
    with correct feature names matching the trained model.
    Fixes: 'Model is missing inputs' schema error
    """
    transformed = vectorizer.transform(comments)
    return pd.DataFrame(
        transformed.toarray(),
        columns=vectorizer.get_feature_names_out()
    )

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {"message": "Welcome to our FastAPI"}


@app.post("/predict")
def predict(request: PredictRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        preprocessed_comments = [preprocess_comment(c) for c in request.comments]
        dense_comments = transform_to_dataframe(preprocessed_comments)
        predictions = model.predict(dense_comments).tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(request.comments, predictions)
    ]


@app.post("/predict_with_timestamps")
def predict_with_timestamps(request: PredictWithTimestampsRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        comments = [item.text for item in request.comments]
        timestamps = [item.timestamp for item in request.comments]
        preprocessed_comments = [preprocess_comment(c) for c in comments]
        dense_comments = transform_to_dataframe(preprocessed_comments)
        predictions = model.predict(dense_comments).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]


@app.post("/generate_chart")
def generate_chart(request: GenerateChartRequest):
    if not request.sentiment_counts:
        raise HTTPException(status_code=400, detail="No sentiment counts provided")
    try:
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(request.sentiment_counts.get('1', 0)),
            int(request.sentiment_counts.get('0', 0)),
            int(request.sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.post("/generate_wordcloud")
def generate_wordcloud(request: GenerateWordCloudRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        preprocessed_comments = [preprocess_comment(c) for c in request.comments]
        text = ' '.join(preprocessed_comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post("/generate_trend_graph")
def generate_trend_graph(request: List[SentimentDataItem]):
    if not request:
        raise HTTPException(status_code=400, detail="No sentiment data provided")
    try:
        data_list = []
        for item in request:
            if hasattr(item, "model_dump"):
                data_list.append(item.model_dump())
            else:
                data_list.append(item.dict())
                
        df = pd.DataFrame(data_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        df['sentiment'] = df['sentiment'].astype(str).str.replace(r'\.0$', '', regex=True)

        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        target_cols = ['-1', '0', '1']
        for col in target_cols:
            if col not in monthly_percentages.columns:
                monthly_percentages[col] = 0.0

        monthly_percentages = monthly_percentages[target_cols]

        sentiment_labels = {'-1': 'Negative', '0': 'Neutral', '1': 'Positive'}
        colors = {'-1': 'red', '0': 'gray', '1': 'green'}

        plt.figure(figsize=(12, 6))
        for sentiment_value in target_cols:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.ylim(-5, 105)
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")
        
    except Exception as e:
        plt.close()
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")

# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import uvicorn
    # 🌟 CRITICAL FIX: Explicitly serve directly onto container port 5000
    uvicorn.run(app, host='0.0.0.0', port=5000)