# Personal Knowledge Assistant (PKA)

A full-stack application that ingests notes, PDFs, books, or research papers and allows you to query them using an Ollama LLM model.

---

## Project Structure

```
.
├── app/             # Streamlit Frontend
├── backend_api/     # FastAPI Backend
├── ingest/          # Data ingestion scripts
├── data/            # Uploaded documents and processed data
├── docker-compose.yml
└── README.md
```

---

## Services

- **Ollama** – Runs the LLM model (`llama3`).
- **Backend API** – FastAPI service for handling queries and ingestion.
- **Ingest Service** – Prepares documents for querying.
- **Streamlit Frontend** – User interface to interact with the assistant.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed.
- Minimum 8GB RAM recommended for `llama3`.

---

## Setup & Run

### 1. Clone Repository

```bash
git clone https://github.com/sufyanaslam2728/personal_knowledge_assistant.git
cd personal_knowledge_assistant
```

### 2. Environment Variables

Create a `.env` file in the project root:

```bash
# Example values
# Embedding model (fast + good default)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Where to save FAISS + metadata INSIDE the container
INDEX_PATH=/app/data/indices/faiss.index
STORE_PATH=/app/data/indices/store.pkl

# Optional: tweak chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100
BACKEND_URL=http://backend_api:8000/query

OLLAMA_URL=http://ollama:11434
DATA_PATH=/app/data
MODEL_NAME=llama3
```

### 3. Build & Start Services

```bash
docker compose up --build
```

This will:

- Start Ollama and pull the `llama3` model.
- Start backend API at [http://localhost:8000](http://localhost:8000)
- Start Streamlit frontend at [http://localhost:8501](http://localhost:8501)

---

## Data Ingestion

To ingest your documents (PDFs, TXT, MD, etc.), place them inside the `data` folder, then run:

```bash
docker compose run ingest python -m ingest.cli --path data/raw
```

This will process documents and store embeddings for LLM queries.

---

## Useful Commands

### Rebuild a Specific Service

```bash
docker compose build backend_api
docker compose up backend_api
docker compose up --build backend_api
```

### Manually Pull Ollama Model

```bash
docker exec -it ollama ollama pull llama3
```

---

## Endpoints

- **Frontend UI** – [http://localhost:8501](http://localhost:8501)
- **API Docs** – [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Volumes

- `ollama` – Stores pulled LLM models.
- `./data` – Stores ingested document data.

---

## Stopping the Application

```bash
docker compose stop
```
