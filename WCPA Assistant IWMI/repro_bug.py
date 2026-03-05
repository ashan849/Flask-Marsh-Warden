
import os
import pickle
import logging
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Mock EnsembleRetriever if needed, or just import it from the file
# For now, let's just test BM25 and FAISS directly.

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_FILE = "pdf_index_enhanced1.pkl"

def test_retrieval():
    if not os.path.exists(INDEX_FILE):
        print(f"Index file {INDEX_FILE} not found!")
        return

    print(f"Loading index {INDEX_FILE}...")
    with open(INDEX_FILE, "rb") as f:
        data = pickle.load(f)
    
    documents = data["documents"]
    embeddings = data["embeddings"]
    texts = [doc.page_content for doc in documents]
    
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    
    print("Building FAISS...")
    faiss_index = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=lambda x: model.encode(x, normalize_embeddings=True),
        metadatas=[doc.metadata for doc in documents]
    )
    faiss_retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    
    print("Building BM25...")
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    
    query = "What are the rules for sanctuaries?"
    
    print(f"Testing FAISS with query: {query}")
    try:
        docs = faiss_retriever.invoke(query)
        print(f"FAISS success: found {len(docs)} docs")
    except Exception as e:
        print(f"FAISS FAILED: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"Testing BM25 with query: {query}")
    try:
        docs = bm25_retriever.invoke(query)
        print(f"BM25 success: found {len(docs)} docs")
    except Exception as e:
        print(f"BM25 FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_retrieval()
