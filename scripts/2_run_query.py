import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA

# --- 1. Configuration ---
VECTOR_DB_PATH = "models/chroma_db"
LLM_MODEL_NAME = "mistral-large-latest" 

# --- 2. Initialization ---
def initialize_rag_components():
    """Loads the vector store and sets up the RAG chain."""
    
    # CRUCIAL: Check for API Key (set via 'export MISTRAL_API_KEY="..."' in terminal)
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY environment variable not set. Please set it before running.")
    
    print("Loading embedding model and vector store...")
    
    # 2a. Load the same embedding model used for ingestion
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # 2b. Load the persistent ChromaDB index
    vector_store = Chroma(
        persist_directory=VECTOR_DB_PATH, 
        embedding_function=embeddings
    )
    
    # 2c. Initialize the LLM (using Mistral for high performance)
    llm = ChatMistralAI(model=LLM_MODEL_NAME, temperature=0) # temperature=0 for factual responses
    
    # 2d. Create the Retriever
    # k=3 retrieves the top 3 most relevant chunks from the documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2e. Create the RAG Chain
    # RetrievalQA is a simple chain that handles retrieval and generation
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'stuff' means all retrieved context is stuffed into the prompt
        retriever=retriever,
        return_source_documents=True # Allows us to show the source for verification
    )
    
    return qa_chain

# --- 3. Execution ---
def run_query(qa_chain, question):
    """Runs a single question through the RAG pipeline."""
    print(f"\n--- Running Query: {question} ---")
    
    # Execute the RAG chain
    result = qa_chain.invoke({"query": question})
    
    # Display the result
    print("\n✅ GENERATED ANSWER:")
    print(result['result'])
    
    print("\n📚 SOURCE DOCUMENTS (RAG Context for Factual Verification):")
    for i, doc in enumerate(result['source_documents']):
        # Metadata 'source' comes from the document loader (e.g., the PDF file name)
        print(f"[{i+1}] Source: {doc.metadata.get('source', 'N/A')}")
        print(f"    Content Snippet: {doc.page_content[:100]}...")
        
    print("-------------------------------------------------")


# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        qa_pipeline = initialize_rag_components()
        
        # Define realistic test questions related to your manuals
        test_questions = [
            "What is the maximum output wattage for the Model Z speaker?",
            "Can I use the included charger for outdoor use according to the manual?",
            "What specific error code means a low-power condition?"
        ]
        
        for q in test_questions:
            run_query(qa_pipeline, q)

    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An error occurred during query execution: {e}")
        print("Tip: Ensure 'models/chroma_db' exists (run 1_ingest_data.py first).")
