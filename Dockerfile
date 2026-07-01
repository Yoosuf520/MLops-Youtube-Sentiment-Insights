# 1. Match your local Python 3.10 virtual environment
FROM python:3.10-slim-buster

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements first to take advantage of Docker caching layers
COPY requirements.txt /app/

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corporate data packages into the container
RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet

# 5. Copy the rest of your local application files
COPY . /app/

# 6. Open up the network port for your Chrome Extension to connect
EXPOSE 8000

# ── FIXED: Changed main.app to main:app ──
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]