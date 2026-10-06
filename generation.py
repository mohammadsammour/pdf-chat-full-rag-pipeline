from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama"
)

conversations = {}

SYSTEM_PROMPT = """You are answering questions about a document. Use only the context below.
If the answer is not supported by the context, say you don't know.
Cite the relevant page numbers, file name and chunk number.
The citation must be beside the answer, not in an isolated section."""

def get_messages(chat_id):
    if chat_id not in conversations:
        conversations[chat_id] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    return conversations[chat_id]

def build_context(results):
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = []
    for doc, meta in zip(docs, metadatas):
        page = meta.get("page", "unknown")
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", "unknown")

        context.append(
            f"page: {page}\n"
            f"source: {source}\n"
            f"chunk: {chunk}\n"
            f"content: {doc}"
        )

    return "\n\n".join(context)

def generate_answer(chat_id, question, context):
    messages = get_messages(chat_id)

    user_prompt = f"""

Chat History: 
{messages}    

Context:
{context}

Question:
{question}
"""

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    chat_completion = client.chat.completions.create(
        model="qwen3:8b",
        messages=messages
    )

    answer = chat_completion.choices[0].message.content

    messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer

def delete_chat_history(chat_id):
    conversations.pop(chat_id, None)