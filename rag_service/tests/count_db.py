import chromadb
from pathlib import Path

DB_PATH = Path(r"c:\Users\a2231\Desktop\RAG\chroma_db")
COLLECTION_NAME = "multimodal_papers"

try:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Total entries in '{COLLECTION_NAME}': {collection.count()}")
except Exception as e:
    print(f"Error accessing ChromaDB: {e}")
