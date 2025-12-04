import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

# Import components from the scripts folder (Assuming we structure imports correctly later)
# For simplicity, we redefine initialization here or move the logic into a shared module.
# For production, we'd put this initialization in a shared module.
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA

# --- Configuration (Must match scripts/1_ingest_data.py) ---
VECTOR_DB_PATH = "models/chroma_db"
LLM_MODEL_NAME = "mistral-large-latest" 

# --- API Initialization ---
app = FastAPI(title="RAG Product Q&A Service", version="1.0")

# --- Pydantic Models for Request/Response ---
class QueryRequest(BaseModel):
    """Defines the structure of the incoming user query."""
    question: str
    
class SourceDocument(BaseModel):
    """Defines the structure for a retrieved source document."""
    source: str
    content: str

class QueryResponse(BaseModel):
    """Defines the structure of the API response."""
    answer: str
    sources: List[SourceDocument]
    model: str = LLM_MODEL_NAME

# --- RAG Components (Global to load only once) ---
qa_chain = None

def initialize_qa_chain():
    """Initializes and returns the RAG RetrievalQA chain."""
    global qa_chain
    if qa_chain is not None:
        return qa_chain
        
    print("Initializing RAG components...")
    
    # Check for API Key
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY environment variable not set.")
    
    # Check for DB existence
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_DB_PATH}. Run 1_ingest_data.py first.")

    # Load components
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH, 
        embedding_function=embeddings
    )
    llm = ChatMistralAI(model=LLM_MODEL_NAME, temperature=0)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    print("RAG components initialized successfully.")
    return qa_chain

# --- FastAPI Hooks ---
@app.on_event("startup")
async def startup_event():
    """Executed when the FastAPI application starts."""
    try:
        initialize_qa_chain()
    except (ValueError, FileNotFoundError) as e:
        print(f"FATAL ERROR during startup: {e}")
        # In a real scenario, you'd handle shutdown or logging here.

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok", "service": "RAG Q&A", "model": LLM_MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Endpoint to process the user's question and return the RAG answer."""
    global qa_chain
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG service not initialized or missing dependencies.")
    
    try:
        # Invoke the RAG chain
        result = qa_chain.invoke({"query": request.question})
        
        # Format source documents for the response model
        sources = [
            SourceDocument(
                source=doc.metadata.get('source', 'N/A'),
                content=doc.page_content
            )
            for doc in result.get('source_documents', [])
        ]

        return QueryResponse(
            answer=result['result'],
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {e}")

