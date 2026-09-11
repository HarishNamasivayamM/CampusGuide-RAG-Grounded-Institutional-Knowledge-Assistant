# CampusGuide RAG - Grounded Institutional Knowledge Assistant

Multi-domain AI assistant for Illinois Institute of Technology student information. The assistant routes questions across academic policies, tuition and fees, academic calendar dates, and department contacts, then retrieves the most relevant records and generates a concise answer.

This project was built as a practical retrieval workflow rather than a single prompt-only chatbot: it combines domain routing, structured search, hybrid retrieval, reranking, clarification handling, and multiple user interfaces. It can run with Elasticsearch locally or with a bundled in-process search backend for simple public hosting.

## What It Does

- Answers IIT student questions across four domains: `DOCUMENTS`, `TUITION`, `CALENDAR`, and `CONTACTS`
- Uses the configured hosted LLM as an intent router and answer generator
- Retrieves evidence from Elasticsearch-backed indexes and curated local data
- Supports single-domain and multi-domain questions with cross-domain reranking
- Maintains lightweight slot state for follow-up questions about tuition and calendar dates
- Exposes the assistant through CLI, Streamlit, and FastAPI entry points

## Architecture

```text
User question
    |
    v
Configured LLM intent router
    |
    +--> DOCUMENTS  -> hybrid BM25 + vector search + rerank -> answer with sources
    +--> TUITION    -> structured fee search + clarification -> answer formatting
    +--> CALENDAR   -> slot extraction + filtered search + rerank -> date answer
    +--> CONTACTS   -> entity extraction + directory search -> structured response
    |
    v
Single-domain answer or cross-domain answer synthesis
```

### Core Components

| Area          | Implementation                                                                           |
| ------------- | ---------------------------------------------------------------------------------------- |
| Routing       | `app/router/router.py` classifies each query into IIT information domains using the configured LLM |
| Orchestration | `app/core/orchestrator.py` owns dispatch, multi-domain retrieval, and state handoff    |
| Retrieval     | Domain handlers combine Elasticsearch search, filters, embeddings, and reranking         |
| Interfaces    | `app/chat.py`, `app/streamlit_app.py`, and `app/api.py`                            |
| Deployment    | Local Elasticsearch/FastAPI, or Streamlit Community Cloud with the bundled local backend |

## Domains Covered

### Academic Policies

Handles academic procedures and policy questions such as transcripts, grading, registration, course withdrawal, hardship withdrawal, health insurance, graduation, academic standing, and related student policy documents.

Retrieval path:

- query preparation and rewrite
- hybrid search over policy chunks
- BM25 + vector retrieval
- Reciprocal Rank Fusion
- cross-encoder reranking
- Configured LLM answer generation with source links

### Tuition and Fees

Handles billing-rate questions such as tuition amounts, mandatory fees, program-specific fees, school-specific costs, academic year rates, and per-credit/per-semester fee queries.

Retrieval path:

- regex-based field extraction
- structured Elasticsearch filters
- clarification when school/year/fee type is ambiguous
- Configured LLM answer formatting over retrieved fee records

### Academic Calendar

Handles dates and deadlines such as semester start/end dates, registration deadlines, holidays, breaks, exams, grade submission, and commencement-related dates.

Retrieval path:

- slot extraction for term, event, month, and date context
- term carryover from conversation history
- clarification for underspecified date questions
- calendar search with filters and reranking

### Contacts

Handles department, office, faculty/staff, email, phone, location, and directory-style contact questions.

Retrieval path:

- entity extraction
- filtered contact search
- deterministic formatted response

## Tech Stack

| Layer         | Tools                                           |
| ------------- | ----------------------------------------------- |
| Language      | Python                                          |
| API           | FastAPI, Uvicorn                                |
| UI            | Streamlit                                       |
| Search        | Elasticsearch                                   |
| Retrieval     | BM25, kNN vector search, Reciprocal Rank Fusion |
| Reranking     | `cross-encoder/ms-marco-MiniLM-L-6-v2`        |
| Embeddings    | `intfloat/e5-large-v2`                        |
| LLM           | Groq via the OpenAI-compatible API (Azure OpenAI remains supported) |
| Data handling | CSV, JSON, NDJSON, openpyxl                     |
| Deployment    | Render                                          |

## Repository Structure

```text
CampusGuide-RAG-Grounded-Institutional-Knowledge-Assistant/
|-- app/
|   |-- api.py                 # FastAPI wrapper
|   |-- chat.py                # CLI entry point
|   |-- streamlit_app.py       # Streamlit chat UI
|   |-- comparison_app.py      # Side-by-side model comparison UI
|   |-- core/
|   |   `-- orchestrator.py    # Shared routing and dispatch logic
|   |-- router/
|   |   `-- router.py          # Configured LLM domain router
|   |-- handlers/              # Domain handlers
|   |-- domains/               # Domain-specific pipeline and search logic
|   `-- common/                # Elasticsearch, LLM, reranking, retrieval utilities
|-- data/
|   |-- curated/               # Tuition fee records
|   |-- processed/             # Calendar and policy chunks
|   `-- raw/                   # Contact data
|-- scripts/                   # Ingestion scripts
|-- render.yaml                # Render deployment config
|-- requirements.txt
`-- .env.example
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/CampusGuide-RAG-Grounded-Institutional-Knowledge-Assistant.git
cd CampusGuide-RAG-Grounded-Institutional-Knowledge-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

For the current Groq setup, use:

```text
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=qwen/qwen3.8-27b
```

For local Elasticsearch mode, also configure:

```text
ES_URL=...
ES_USER=...
ES_PASS=...
```

Azure OpenAI remains supported with:

```text
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
```

### Free public Streamlit deployment

Streamlit Community Cloud can run the bundled data without Elasticsearch. In the app's Cloud secrets, use TOML like this (replace the placeholder with your own key):

```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your_groq_key"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3.8-27b"
SEARCH_BACKEND = "local"
```

Deploy `app/streamlit_app.py` from the repository root. This mode uses the checked-in tuition, calendar, contacts, and policy files, so no Elasticsearch URL or password is needed.

## Running the App

### CLI

```bash
python -m app.chat
```

### Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### FastAPI Server

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Chat request:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "When is the add/drop deadline for fall?",
    "history": [],
    "tuition_state": {},
    "calendar_state": {}
  }'
```

## API Contract

### `GET /health`

Returns:

```json
{
  "status": "ok",
  "model": "group7"
}
```

### `POST /chat`

Request:

```json
{
  "query": "How much is tuition for Stuart School?",
  "history": [
    {"role": "user", "content": "Previous user message"},
    {"role": "assistant", "content": "Previous assistant reply"}
  ],
  "tuition_state": {},
  "calendar_state": {}
}
```

Response:

```json
{
  "reply": "string",
  "is_clarification": false,
  "domains": ["TUITION"],
  "tuition_state": {},
  "calendar_state": {}
}
```

## Data and Ingestion

The repository includes curated data files used by the assistant:

- `data/curated/tuition_fees.bulk.ndjson`
- `data/processed/calendar_chunks.json`
- `data/processed/Unstructured data/Unstructured chunks k.json`
- `data/raw/Contacts data.csv`

Ingestion scripts are available under `scripts/` for rebuilding Elasticsearch indexes:

- `scripts/ingest_contacts.py`
- `scripts/reingest_policies.py`
- `scripts/ingest_new_policy_chunks.py`

The app expects Elasticsearch indexes to be available before answering retrieval-backed questions.

## Deployment

The included `render.yaml` deploys the FastAPI app on Render:

```yaml
startCommand: uvicorn app.api:app --host 0.0.0.0 --port $PORT
```

Set the same environment variables from `.env.example` in the Render dashboard.

## Notes and Limitations

- The assistant is scoped to IIT academic information and should not answer unrelated questions.
- Some workflows require a live Elasticsearch deployment and configured LLM credentials.
- Source quality depends on the freshness of the indexed IIT policy, calendar, tuition, and contact data.
- Tuition and calendar flows include clarification handling, but ambiguous questions may still need follow-up.

## Author

Built by
