# SageMaker AI Documentation Assistant 🤖

This is a Retrieval-Augmented Generation (RAG) application that acts as an expert assistant for the Amazon SageMaker Developer Guide. It allows users to ask complex questions and receive accurate, context-aware answers derived strictly from the official 9,500-page AWS manual.

## Architecture

*   **Data Ingestion:** LangChain (`PyPDFLoader` & `RecursiveCharacterTextSplitter`)
*   **Vector Database:** Chroma
*   **Embeddings:** OpenAI (`text-embedding-3-small`)
*   **LLM:** OpenAI (`gpt-4o-mini`)
*   **Backend:** FastAPI
*   **Frontend:** Streamlit

## Local Setup Instructions

### 1. Prerequisites
You must have Python and the `uv` package manager installed.
You also need an OpenAI API key. Create a `.env` file in the root directory and add:
```
OPENAI_API_KEY=sk-your-key-here
```

### 2. Install Dependencies
Initialize the environment and install packages using `uv`:
```bash
uv venv
uv init
uv add fastapi uvicorn pydantic langchain langchain-openai langchain-chroma langchain-community pypdf python-dotenv streamlit requests
```

### 3. Build the Database
*Note: You only need to do this once. It will download the PDF and chunk it into the local vector database.*
```bash
uv run python src/download_pdf.py
uv run python src/ingestion.py
```

### 4. Run the Application
You will need two terminal windows to run both servers simultaneously.

**Terminal 1 (Backend API):**
```bash
uv run uvicorn src.main:app --reload
```

**Terminal 2 (Frontend UI):**
```bash
uv run streamlit run src/frontend.py
```
The application will open automatically in your browser at `http://localhost:8501`.
