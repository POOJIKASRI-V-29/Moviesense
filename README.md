# MovieSense

### NLP-Based Movie Genre Classification, Explainability and Similar Movie Retrieval

[![Live Application](https://img.shields.io/badge/Live%20Application-MovieSense-orange?style=for-the-badge)](https://moviesense-frontend-434435247015.asia-south1.run.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/POOJIKASRI-V-29/Moviesense)
[![Google Cloud](https://img.shields.io/badge/Deployed-Google%20Cloud%20Run-4285F4?style=for-the-badge&logo=googlecloud)](https://cloud.google.com/run)

**MovieSense** is an NLP-powered movie analysis application that predicts movie genres from plot summaries, explains the words influencing its predictions, and retrieves similar movies from the CMU Movie Summary Corpus.

The project implements and compares two multi-label text classification approaches: a traditional **TF-IDF + Logistic Regression** baseline and a fine-tuned **DistilBERT** model. It also incorporates explainable predictions, content-based movie retrieval and metadata-based cast matching.

## Live Application

<https://moviesense-frontend-434435247015.asia-south1.run.app/>

Try MovieSense by entering a movie plot description and exploring its predicted genres, explanations and similar movies.

### Backend APIs

| Service | Endpoint | Documentation |
|---|---|---|
| Baseline API | [MovieSense Baseline](https://moviesense-baseline-434435247015.asia-south1.run.app/) | [Swagger UI](https://moviesense-baseline-434435247015.asia-south1.run.app/docs) |
| DistilBERT API | [MovieSense DistilBERT](https://moviesense-distilbert-434435247015.asia-south1.run.app/) | [Swagger UI](https://moviesense-distilbert-434435247015.asia-south1.run.app/docs) |

**GitHub Repository:** https://github.com/POOJIKASRI-V-29/Moviesense

---

## Key Features

- **Multi-label Genre Classification:** Predicts multiple movie genres from a plot description across 19 mapped genres.
- **Baseline Classification:** Uses TF-IDF feature extraction with Logistic Regression.
- **Transformer-Based Classification:** Uses fine-tuned DistilBERT for contextual text representation and genre prediction.
- **Explainable Predictions:** Highlights influential words using Integrated Gradients for DistilBERT and linear feature attribution for the baseline.
- **Similar Movie Retrieval:** Retrieves movies using TF-IDF-based plot similarity.
- **Cast Matching:** Displays cast information only when a sufficiently reliable match is available in the movie metadata.
- **Genre Confidence Scores:** Displays model scores and per-genre prediction thresholds.
- **Model Comparison:** Supports comparison between traditional machine learning and transformer-based classification.

## Application Overview

The application provides two main workflows:

1. **Analyse a Plot:** Enter a movie plot description to obtain genre predictions, explanations, similar movies and cast-matching results.
2. **Similar to a Title:** Explore movies based on an existing movie title.

The plot analysis interface displays genre scores, influential words, similar movie results and available cast metadata.

## Technology Stack

| Component | Technologies |
|---|---|
| Frontend | React, Vite |
| Backend | Python, FastAPI |
| Baseline Model | TF-IDF, Logistic Regression |
| Deep Learning Model | DistilBERT, PyTorch, Transformers |
| Explainability | Integrated Gradients, Linear Feature Attribution |
| Retrieval | TF-IDF, Cosine Similarity |
| Dataset | CMU Movie Summary Corpus |
| Cloud Platform | Google Cloud Run |
| Containerization | Docker |
| Container Registry | Google Artifact Registry |

## Machine Learning Approaches

### 1. TF-IDF + Logistic Regression

The baseline model converts movie plot summaries into numerical TF-IDF feature representations and applies Logistic Regression for multi-label genre classification.

It provides a traditional machine learning approach for comparison with the transformer-based model.

### 2. DistilBERT

The proposed transformer-based approach uses a fine-tuned DistilBERT model to capture contextual relationships in movie plot descriptions and predict multiple genres.

### 3. Explainability

MovieSense provides model-specific explanations:

- **DistilBERT:** Integrated Gradients-based word attribution.
- **Baseline:** Linear feature attribution derived from TF-IDF features and model coefficients.

These explanations identify words influencing a model's prediction. They do not establish whether the prediction is objectively correct.

### 4. Similar Movie Retrieval

The deployed application uses TF-IDF representations and cosine similarity to retrieve movies with similar plot descriptions.

### 5. Cast Matching

Cast information is retrieved from available movie metadata when a sufficiently reliable match is found. MovieSense does not invent actor names from plot descriptions.

---

## Dataset

MovieSense uses the **CMU Movie Summary Corpus**, which contains movie plot summaries and associated metadata.

The project performs data cleaning, metadata integration, genre mapping and preparation of multi-label classification datasets.

Current dataset processing records include:

| Dataset Statistic | Count |
|---|---:|
| Plot summaries | 42,306 |
| Movie records | 81,741 |
| Records joined with metadata | 42,207 |
| Records after cleaning | 39,292 |
| Mapped genres | 19 |
| Training samples | 27,574 |
| Validation samples | 5,867 |
| Test samples | 5,851 |

*Counts are reported according to the corresponding dataset processing and split records.*

---

## Cloud Deployment

MovieSense Phase 1 deployed two independent model APIs using **Google Cloud Run** in the `asia-south1` (Mumbai) region.

| Service | Model | Deployment Status |
|---|---|---|
| Baseline API | TF-IDF + Logistic Regression | Deployed and validated |
| DistilBERT API | DistilBERT | Deployed and validated |
| Frontend | MovieSense web application | Deployed |

### API Endpoints

Both backend services expose the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service health check |
| `/docs` | GET | Interactive API documentation |
| `/api/predict` | POST | Predict movie genres |
| `/api/recommend` | POST | Retrieve similar movies |

### Health Checks

- [Baseline Health](https://moviesense-baseline-434435247015.asia-south1.run.app/health)
- [DistilBERT Health](https://moviesense-distilbert-434435247015.asia-south1.run.app/health)

### Example API Request

```bash
curl -X POST \
  https://moviesense-baseline-434435247015.asia-south1.run.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"A detective investigates a mysterious disappearance and uncovers a dangerous conspiracy."}'
```

Replace the baseline URL with the DistilBERT URL to use the transformer-based classifier.

### Deployment Validation

During Phase 1 validation:

- Both Cloud Run services passed HTTP 200 health checks.
- Both prediction endpoints returned successful responses.
- Both recommendation endpoints returned successful responses.
- Both services used the shared TF-IDF retrieval encoder.

The deployment was validated using the same test synopsis for both models. A broader held-out evaluation is necessary before making general comparative performance claims.

**Cloud deployment region:** `asia-south1` (Mumbai)

---

## Local Setup

### Prerequisites

- Python 3.10 or later, compatible with the project dependencies.
- Node.js and npm.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/POOJIKASRI-V-29/Moviesense.git
cd Moviesense
```

### 2. Create a Python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r requirements.txt
```

For the machine learning environment, install additional dependencies if required:

```bash
pip install -r requirements-ml.txt
```

### 4. Configure environment variables

Create a local `.env` file if required by the project configuration.

Use `.env.example` as a reference where available.

Never commit API keys, credentials or other secrets.

### 5. Start the backend

From the project root:

```bash
./scripts/run_backend.sh
```

The local backend is configured to run at:

```text
http://127.0.0.1:8001
```

API documentation:

```text
http://127.0.0.1:8001/docs
```

### 6. Start the frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local development URL printed by Vite in your browser.

**Note:** Local execution requires the appropriate dataset files and model artifacts to be available. These may not be included in the Git repository.

---

## Project Structure

```text
MovieSense/
│
├── configs/                 # Project configuration
├── frontend/                # React and Vite application
├── notebooks/               # Research and experimentation notebooks
├── reports/                 # Evaluation reports and experimental outputs
├── scripts/                 # Training, evaluation and execution scripts
├── src/
│   └── moviesense/          # Core application and model implementation
├── tests/                   # Unit and integration tests
│
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── requirements-ml.txt
└── README.md
```

---

## Research and Evaluation

MovieSense evaluates traditional machine learning and transformer-based approaches using multi-label classification metrics.

The evaluation includes:

- Macro-F1 and Micro-F1.
- Precision and Recall.
- Per-genre confidence scores and thresholds.
- Explainability evaluation.
- Similar movie retrieval performance.
- Inference time and model size.

Experimental results must be interpreted according to their respective datasets, splits, seeds and evaluation settings.

---

## Limitations and Future Work

- Further improvement of similar movie retrieval relevance.
- Broader held-out evaluation for stronger comparative conclusions.
- Continued evaluation of metadata-based cast matching.
- Further optimization of model inference and deployment resources.
- Additional improvements to production deployment configuration and access control.

---

## Author

**Pooji Kasri V.**

B.Tech – Computer Science and Engineering

GitHub: [POOJIKASRI-V-29](https://github.com/POOJIKASRI-V-29)

---

## Dataset Attribution

MovieSense uses the CMU Movie Summary Corpus for research and development.

Original dataset publication:

Bamman, D., O'Connor, B., and Smith, N. A. (2013). Learning Latent Personas of Film Characters. *Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)*.

Refer to the original dataset publication and licensing terms for attribution and redistribution requirements.

---

**MovieSense — From Movie Plots to Explainable Genre Predictions.**
