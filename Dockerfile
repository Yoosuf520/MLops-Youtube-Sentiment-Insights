# base image
FROM python:3.12-slim-bookworm

# install uv
RUN pip install uv

# workdir
WORKDIR /app

# copy files into container workspace
COPY . /app

# install dependencies using uv
RUN uv pip install -r requirements.txt --system

# 🌟 CRITICAL FIX: Download NLTK corpora to prevent preprocessing crashes
RUN python -m nltk.downloader stopwords wordnet

# 🌟 CRITICAL FIX: Update expose port from 8000 to 5000 to match AWS and popup.js
EXPOSE 5000

# 🌟 CRITICAL FIX: Modify uvicorn command flags to run directly on port 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]