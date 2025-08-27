from fastapi import FastAPI, HTTPException
import time
from pydantic import BaseModel
import os
from ingest.embedder import Embedder
from ingest.indexer import FaissStore
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Environment variables with defaults
INDEX_PATH = os.getenv("INDEX_PATH", "/app/data/indices/faiss.index")
STORE_PATH = os.getenv("STORE_PATH", "/app/data/indices/store.pkl")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# FastAPI application
app = FastAPI(title="Personal Knowledge Assistant API")

# Initialize components
try:
    embedder = Embedder(EMBEDDING_MODEL)
    store = FaissStore(INDEX_PATH, STORE_PATH)
    store.load()  # Ensure index is preloaded
    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_HOST)
except Exception as e:
    raise RuntimeError(f"Failed to initialize components: {e}")

# Prompt Template for LLM
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant.
Use the context below to answer the question concisely.

Context:
{context}

Question:
{question}

Answer:""",
)
chain = LLMChain(prompt=prompt, llm=llm)

# Request Model
class QueryRequest(BaseModel):
    question: str
    k: int = 4

@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/query")
def query_rag(payload: QueryRequest):
    """
    Perform Retrieval-Augmented Generation (RAG) on user query.
    """
    start_time = time.perf_counter()
    print("Payload", payload)
    try:
        # Embed query and search
        qvec = embedder.embed_query(payload.question)
        results = store.search(qvec, k=payload.k)

        if not results:
            return {"answer": "No relevant context found.", "sources": []}

        # Concatenate top results for context
        context = "\n\n".join([r["text"] for r in results])

        # Generate final answer
        answer = chain.run(context=context, question=payload.question)
        
        total_time = time.perf_counter() - start_time  # ⏱ end timer
        print(f"⏱ Total time taken: {total_time:.2f} seconds")

        return {"answer": answer, "sources": results}

    except Exception as e:
        total_time = time.perf_counter() - start_time  # ⏱ end timer
        print(f"⏱ Total time taken: {total_time:.2f} seconds")
        
        raise HTTPException(status_code=500, detail=str(e))
