

```markdown
# YouTube Comment Sentiment Insights 🚀
> **An End-to-End MLOps Pipeline for Real-Time Video Audience Analytics**

A production-ready MLOps pipeline that tracks and visualizes YouTube audience sentiment in real-time. By leveraging a custom Google Chrome Extension frontend and a containerized FastAPI backend on AWS EC2, the system processes live comment data streams, executes model inference via a strict-schema LightGBM classifier tracked by MLflow, and returns interactive visual dashboards.

---

## 🛠️ System Architecture

```text
┌──────────────────────────┐     POST JSON Payload     ┌───────────────────────────┐
│   Chrome Extension UI    │ ────────────────────────> │   FastAPI Engine (8000)   │
│  (popup.js / popup.html) │ <──────────────────────── │    (Containerized on EC2) │
└──────────────────────────┘    Base64 Chart Strings   └─────────────┬─────────────┘
                                                                     │
                                                           Loads Artifact Vectors
                                                                     ▼
                                                       ┌───────────────────────────┐
                                                       │   MLflow Tracking Node    │
                                                       │    (S3 Weights Registry)  │
                                                       └───────────────────────────┘

```

### Core Execution Workflow

1. **Extraction**: The browser extension extracts raw text comment strings directly from the active YouTube watch link using the **YouTube Data API v3**.
2. **Gateway Processing**: A containerized **FastAPI** service handles cross-origin requests via customized CORS configurations, maps the payload layout, and shifts text structures into dense `pd.DataFrame` arrays.
3. **Inference Suite**: The tabular feature vectors match enforcement model signatures precisely, allowing the logged **LightGBM** classifier to output sentiment targets (`-1`: Negative, `0`: Neutral, `1`: Positive).
4. **Visual Synthesis**: Headless backend graphic buffers (`matplotlib` run over an `Agg` context layer) render isolated analytics assets dynamically, returning base64-encoded strings back to the user interface panel.

---

## 🏗️ Production Infrastructure & MLOps Practices

### 📂 Data Version Control (DVC)

* **The Problem**: Git is built for lightweight text code changes; committing heavy binary artifacts like dataset iterations, `tfidf_vectorizer.pkl`, or model checkpoint weights bloats repository size and destroys commit histories.
* **The Solution**: DVC intercepts binary mutations. It tracks small, explicit pointer files (`.dvc`) natively inside Git, while streaming the underlying heavy asset files directly to an isolated **AWS S3 bucket store**. This guarantees 100% baseline reproducibility.

### 🔑 YouTube Data API v3 Setup (GCP Console)

To access live comment streams, a free connection token must be fetched from the Google Cloud Platform console:

1. Initialize a workspace project on the [GCP Console](https://www.google.com/search?q=https://console.cloud.google.com/).
2. Head to the **API Library**, run a query for **YouTube Data API v3**, and toggle it to **Enable**.
3. Under the **Credentials** dashboard, select **+ Create Credentials** -> **API Key**.
4. Save the generated key directly into your extension's local environment files.

> 💡 *Note: The YouTube Data API v3 tier includes a free, complimentary daily allocation quota of **10,000 units**, which easily handles thousands of testing requests without cost.*

### 🔐 Identity & Access Management (IAM) Configuration

Automated deploy engines require explicit access privileges. Create a programmatic IAM User entity named `mlops-pipeline-runner` inside AWS and map these exact managed policy scopes:

* `AmazonEC2ContainerRegistryFullAccess`
* `AmazonS3FullAccess`
* `AmazonEC2FullAccess`

### 📦 Elastic Container Registry (ECR) Image Hosting

We push our production images to Amazon ECR. This provides a secure, cloud-hosted private registry for our compiled container layers. By decoupling the image store, the automated actions workflow can securely pull down clean builds directly onto our EC2 node **publicly anywhere and anytime**, keeping deployments entirely self-contained.

---

## 📊 Showcasing MLflow UI Experiments

We use MLflow to systematically track hyperparameters, validation logs, execution parameters, and model schemas during training loops.

### Experiment Management Interface

The server runs independent tracking services inside an isolated virtual workspace (`tmux` background session) on port `5000` to register historical pipeline parameters permanently:

> ### 📊 Live MLflow Tracking Dashboard
You can view the live model training registry, historical hyperparameter runs, and strict schema enforcements directly via the public tracking node:

 **[Launch Live MLflow Experiments UI](http://3.110.185.110:5000)**

* **Model Registry & Run Parameters**:

* **Strict Signature Schema Enforcement**:


---

## 🐋 Dockerization & CLI Operations

The application leverages a caching-optimized Debian build profile to isolate environmental dependencies and protect LightGBM system execution loops.

```bash
# Compile and build the container application layers locally
docker build -t mlproject:latest .

# Launch the FastAPI service microservice detached on port 8000
docker run -d -p 8000:8000 --name youtube-sentiment-app mlproject:latest

# Monitor active runtime health and container lifecycle statuses
docker ps

# Inspect raw internal exception stacks and runtime print statements
docker logs youtube-sentiment-app --tail 50

```

### AWS EC2 Firewall Setup (Security Groups)

Your active security group parameters require explicit configuration rules to navigate web traffic past the cloud instance gateway:

| Protocol | Port Range | Source Target | Purpose |
| --- | --- | --- | --- |
| **SSH** | `22` | `My IP (X.X.X.X/32)` | Secure remote configuration access |
| **Custom TCP** | `8000` | `Anywhere (0.0.0.0/0)` | **FastAPI Inbound Traffic**: Processes extension comment streams |
| **Custom TCP** | `5000` | `Anywhere (0.0.0.0/0)` | **MLflow Web UI Portal**: Exposes the experiment dashboard |

---

## 🔄 GitHub Actions CI/CD Pipeline

Every code update pushed to the `main` branch fires our automated build and deploy workflow (`.github/workflows/cicd.yaml`):

```text
┌────────────────┐      Syntax Sanity       ┌────────────────┐      Relaunch Pod
│  git push main │ ───────────────────────> │  Docker Build  │ ───────────────────────> Live App Up
└────────────────┘      Check Passed        │  & Push to ECR │      via Remote SSH      on AWS EC2
                                            └────────────────┘

```

1. **Syntax Linting**: Runs a `py_compile` syntax pre-check directly on `main.py` to intercept and stop builds if structural code typos exist.
2. **Compilation & Shipment**: Logs into the Amazon ECR node, compiles the layered image stack, bakes in your static `tfidf_vectorizer.pkl` feature map array, and pushes the container under the `:latest` tag registry.
3. **Automated Live Reload**: Opens a secure SSH bridge to the destination EC2 node, authenticates with ECR, brings down the obsolete application container, pulls the fresh image layer, and spins up the live service instantly.

---

##  Core API Specification

### `POST /predict`

Processes inbound comment arrays, performs dense matrix shape data framing on the fly, and routes inputs into the evaluation model graph.

* **Extension Data Request Format (JSON)**:
```json
{
  "comments": [
    {
      "text": "This pipeline MLOps project is implemented flawlessly!",
      "published_at": "2026-07-02T15:15:00Z"
    }
  ]
}

```


* **FastAPI Response Payload Output (JSON)**:
```json
{
  "success": true,
  "predictions": [
    {
      "comment": "This pipeline MLOps project is implemented flawlessly!",
      "sentiment": "1",
      "timestamp": "2026-07-02T15:15:00Z",
      "hour": 15
    }
  ],
  "metrics": {
    "total_comments": 1,
    "positive": 1,
    "negative": 0,
    "neutral": 0
  },
  "visualizations": {
    "sentiment_donut_chart": "iVBORw0KGgoAAAANSUhEUgAA...",
    "wordcloud_chart": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  "time_distribution_workload": { "15": 1 }
}



## 🛡️ License

This project is open-source and released under the terms of the [MIT License](LICENSE).
