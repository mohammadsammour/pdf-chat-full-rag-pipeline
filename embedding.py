from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def embed_documents(texts):

    doc_embeddings = EMBEDDING_MODEL.embed_documents(texts)

    return doc_embeddings