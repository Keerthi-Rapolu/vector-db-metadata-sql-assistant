import os
import chromadb
from chromadb.config import Settings

def get_client():
    persist_dir = os.getenv("CHROMA_DIR", "chroma_store")
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

def get_collection(client):
    name = os.getenv("CHROMA_COLLECTION", "metadata_dictionary")
    return client.get_or_create_collection(name=name)

def reset_collection(client):
    name = os.getenv("CHROMA_COLLECTION", "metadata_dictionary")
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name)
