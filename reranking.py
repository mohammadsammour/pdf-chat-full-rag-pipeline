from sentence_transformers import CrossEncoder

RERANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    device="cpu",
)


def rerank_results(question, results, top_k=5):
    """Rerank Chroma results and return the best chunks."""
    texts = results["documents"][0]

    if not texts:
        return results

    ranked = RERANKER.rank(
        question,
        texts,
        top_k=top_k,
        batch_size=8,
    )

    indices = [item["corpus_id"] for item in ranked]

    return {
        "documents": [[texts[i] for i in indices]],
        "metadatas": [[results["metadatas"][0][i] for i in indices]],
        "ids": [[results["ids"][0][i] for i in indices]],
        "distances": [[results["distances"][0][i] for i in indices]],
        "rerank_scores": [[float(item["score"]) for item in ranked]],
    }