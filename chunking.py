from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter

def fixed_size_word_chunking(text, page,file_name,chunk_size=500, overlap=50):
    """This Function Return List of Documents"""
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for chunk_number,start in enumerate(range(0, len(words), step),start=1):
        chunk = words[start:start + chunk_size]
        document = Document(
            page_content= " ".join(chunk),
            metadata={"page": page,
                      "source":file_name,
                      "chunk":chunk_number
                      }
        )
        chunks.append(document)

    return chunks

def recursive_character_chunking(text,page,file_name,separators = ["\n\n","\n"," ",""],chunk_size=500,overlap=50):
    """This Function Return List of Documents"""
    text_splitter = RecursiveCharacterTextSplitter(
    separators=separators,
    chunk_size=chunk_size,
    chunk_overlap=overlap,
    length_function=len,
    )

    documents = text_splitter.create_documents([text],
                                          metadatas=[{"page": page,
                                                      "source":file_name}])
    for chunk_number,doc in enumerate(documents):
        doc.metadata["chunk"] = chunk_number
    return documents

def character_chunking(text, page, file_name, chunk_size=1000, overlap=200):
    """Split text by character and keep the source and page number."""
    if not text or not text.strip():
        return []

    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    chunks = splitter.create_documents(
        [text],
        metadatas=[{"page": page, "source": file_name}],
    )

    for chunk_number, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = chunk_number

    return chunks
    
    
    
