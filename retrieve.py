import faiss
import json
import numpy as np

from sentence_transformers import SentenceTransformer


def retrieve(query, top_k=3):

    # Load FAISS index
    index = faiss.read_index("vectorstore/index.faiss")

    # Load document chunks
    with open("vectorstore/chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Convert user query into an embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize query embedding
    faiss.normalize_L2(query_embedding)

    # Search for similar chunks
    scores, indices = index.search(query_embedding, top_k)

    results = []

    # Retrieve matching chunks
    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":

    query = input("\nAsk a question about your syllabus: ")

    results = retrieve(query)

    print("\nRelevant Results:\n")

    for i, result in enumerate(results, start=1):

        print(f"Result {i}")
        print(f"Page: {result['page']}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(result["text"])
        print("-" * 60)