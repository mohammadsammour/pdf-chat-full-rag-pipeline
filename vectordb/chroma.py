import chromadb
from pathlib import Path
VECTORDB_DIR = Path(__file__).resolve().parent
CHROMA_DIR = VECTORDB_DIR / "chroma_data"

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(name="Research_Papers_character")

def add_to_collection(documents,embeddings,ids,metadatas):
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
def upsert_collection(documents,embeddings,ids,metadatas):
    collection.upsert(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
def query_collection(query_embedding,n_results=5):
    results = collection.query(
        query_embeddings = query_embedding,
        n_results=n_results
    )
    return results