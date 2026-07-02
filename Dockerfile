# 1. Use an official, stable Debian-based Python runtime image
FROM python:3.10-slim-bullseye

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install LightGBM OS system dependencies (OpenMP) to prevent libgomp crashes
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only requirements first to take advantage of Docker caching layers
COPY requirements.txt /app/

# 5. Install Python dependencies and pre-download NLTK corporate asset data packages
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet

# 6. Copy the rest of your local application files (including main.py and tfidf_vectorizer.pkl)
COPY . /app/

# 7. Expose the port FastAPI will run on
EXPOSE 8000

# 8. Command to start the application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]