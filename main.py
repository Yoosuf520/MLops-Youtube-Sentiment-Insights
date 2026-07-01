import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ──
class Comment(BaseModel):
    text: str
    timestamp: Optional[str] = None

class PredictRequest(BaseModel):
    comments: List[str]

class PredictWithTimestampRequest(BaseModel):
    comments: List[Comment]

class ChartRequest(BaseModel):
    sentiment_counts: dict

class WordCloudRequest(BaseModel):
    comments: List[str]

class SentimentDataRequest(BaseModel):
    sentiment_data: List[dict]

# ── Preprocessing ──
def preprocess_comment(comment):
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
        print(f"Error in preprocessing: {e}")
        return comment

# ── Load Model ──
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # This tells the container to step out to the EC2 host gateway on port 5000
    mlflow.set_tracking_uri("http://172.17.0.1:5000")
    
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer(
    "youtube_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl"
)

# Extract vocabulary column headers once to save computation time
feature_names = vectorizer.get_feature_names_out()

# ── Routes ──
@app.get('/')
def home():
    return {"message": "Welcome to our FastAPI"}

@app.post('/predict')
def predict(request: PredictRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        preprocessed = [preprocess_comment(c) for c in request.comments]
        transformed = vectorizer.transform(preprocessed)
        dense = transformed.toarray()
        
        # ── FIX: Convert raw matrix into DataFrame with exact vocabulary feature names ──
        input_df = pd.DataFrame(dense, columns=feature_names)
        
        predictions = model.predict(input_df).tolist()
        return [{"comment": c, "sentiment": s} for c, s in zip(request.comments, predictions)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/predict_with_timestamps')
def predict_with_timestamps(request: PredictWithTimestampRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        comments = [item.text for item in request.comments]
        timestamps = [item.timestamp for item in request.comments]
        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)
        dense = transformed.toarray()
        
        # ── FIX: Convert raw matrix into DataFrame with exact vocabulary feature names ──
        input_df = pd.DataFrame(dense, columns=feature_names)
        
        predictions = [str(p) for p in model.predict(input_df).tolist()]
        return [{"comment": c, "sentiment": s, "timestamp": t}
                for c, s, t in zip(comments, predictions, timestamps)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post('/generate_chart')
def generate_chart(request: ChartRequest):
    try:
        sizes = [
            int(request.sentiment_counts.get('1', 0)),
            int(request.sentiment_counts.get('0', 0)),
            int(request.sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        labels = ['Positive', 'Neutral', 'Negative']
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=140,
                textprops={'color': 'w'})
        plt.axis('equal')
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")

@app.post('/generate_wordcloud')
def generate_wordcloud(request: WordCloudRequest):
    try:
        preprocessed = [preprocess_comment(c) for c in request.comments]
        text = ' '.join(preprocessed)
        wordcloud = WordCloud(
            width=800, height=400,
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
        raise HTTPException(status_code=500, detail=f"Wordcloud failed: {str(e)}")

@app.post('/generate_trend_graph')
def generate_trend_graph(request: SentimentDataRequest):
    try:
        df = pd.DataFrame(request.sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        for s in [-1, 0, 1]:
            if s not in monthly_percentages.columns:
                monthly_percentages[s] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]
        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        for s in [-1, 0, 1]:
            plt.plot(monthly_percentages.index,
                     monthly_percentages[s],
                     marker='o', linestyle='-',
                     label=sentiment_labels[s],
                     color=colors[s])
        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage (%)')
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
        raise HTTPException(status_code=500, detail=f"Trend graph failed: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)