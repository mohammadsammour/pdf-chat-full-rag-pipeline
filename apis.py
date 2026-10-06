from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from file_handling import extract_text
from chunking import recursive_character_chunking
from embedding import embed_documents
from vectordb.chroma import add_to_collection,query_collection,collection
from generation import build_context,generate_answer,SYSTEM_PROMPT,delete_chat_history
from chunking import character_chunking
from reranking import rerank_results

app = FastAPI()
@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    pages = extract_text(file.file)

    documents = []
    ids = []

    for page in pages:
        chunks = character_chunking(
                    text=page["text"],
                    page=page["page"],
                    file_name=file.filename,
                    chunk_size=1000,
                    overlap=200,
                    )
        print(type(chunks))
        print(chunks)
        documents.extend(chunks)
        
        

        ids.extend([
            f"{file.filename}_page_{page['page']}_chunk_{j}"
            for j in range(len(chunks))
        ])

    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    embeddings = embed_documents(texts)
    print("Adding:", len(documents))
    print("Before:", collection.count())
    add_to_collection(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print("After:", collection.count())

    return {
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(documents)
    }

class ChatRequest(BaseModel):
    question: str
    chat_id : str

@app.post("/chat")
async def chat(request: ChatRequest):
    question_embedding = embed_documents([request.question])

    results = query_collection(query_embedding=question_embedding,
                               n_results=50,
                               )

    results = rerank_results(question=request.question,
                             results=results,
                             top_k=5,)

    context = build_context(results)

    answer = generate_answer(
        chat_id=request.chat_id,
        question=request.question,
        context=context
    )

    return {
        "question": request.question,
        "answer": answer
    }

@app.delete("/chat/{chat_id}")
async def delete_chat(chat_id: str):
    delete_chat_history(chat_id)
    return {"message": "Chat deleted",
            "chat_id":chat_id}
