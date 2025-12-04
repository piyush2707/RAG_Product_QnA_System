import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
DATA_PATH = "data/"
VECTOR_DB_PATH = "models/chroma_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Function Definitions ---
def load_documents(data_path):
    """Loads all PDF documents from the specified data path."""
    print(f"Loading documents from {data_path}...")
    
    # Use DirectoryLoader to load all PDF files in the directory
    loader = DirectoryLoader(
        data_path,
        glob="**/*.pdf",  # Target only PDF files
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} pages/documents.")
    return docs

def split_documents(docs):
    """Splits documents into smaller, overlapping chunks."""
    print("Splitting documents into context chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(docs)
    print(f"Created {len(texts)} chunks for indexing.")
    return texts

def create_vector_store(texts, db_path):
    """Creates embeddings and persists the vector store using ChromaDB."""
    print("Creating embeddings and storing in vector database...")
    
    # Use a widely recognized, robust open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create the Chroma DB instance and persist the data
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_path
    )
    db.persist()
    print(f"Vector store successfully created and persisted at {db_path}")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure necessary directory structure exists
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    
    # Simple check to ensure the user has placed some data
    if not any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        print("\nERROR: Please place your product manuals (.pdf files) into the 'data/' directory first.")
    else:
        documents = load_documents(DATA_PATH)
        chunks = split_documents(documents)
        create_vector_store(chunks, VECTOR_DB_PATH)
