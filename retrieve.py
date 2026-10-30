
import faiss
import json
import numpy as np
import re

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ==========================================
# LOAD RESOURCES ONCE
# ==========================================

print("Loading DocuMind resources...")

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("vectorstore/index.faiss")

with open("vectorstore/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)


# ==========================================
# INITIALIZE BM25
# ==========================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


tokenized_chunks = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

print("FAISS and BM25 loaded successfully!")


# ==========================================
# HYBRID RETRIEVAL
# ==========================================

def retrieve(query, top_k=5):

    query_lower = query.lower()

    # --------------------------------
    # Identify course from query
    # --------------------------------

    target_course = None

    if "logic design" in query_lower:
        target_course = "CS2302"

    # --------------------------------
    # Identify requested unit
    # --------------------------------

    unit_match = re.search(
        r"\bunit\s+(\d+)\b",
        query_lower
    )

    target_unit = None

    if unit_match:
        target_unit = int(unit_match.group(1))

    # --------------------------------
    # Semantic Search
    # --------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, semantic_indices = index.search(
        query_embedding,
        min(30, len(chunks))
    )

    # --------------------------------
    # BM25 Search
    # --------------------------------

    keyword_scores = bm25.get_scores(tokenize(query))

    keyword_indices = np.argsort(keyword_scores)[::-1][:30]

    # --------------------------------
    # Combine Rankings
    # --------------------------------

    combined_scores = {}

    for rank, idx in enumerate(semantic_indices[0]):

        if idx == -1:
            continue

        combined_scores[idx] = combined_scores.get(idx, 0) + (
            1 / (60 + rank + 1)
        )

    for rank, idx in enumerate(keyword_indices):

        combined_scores[idx] = combined_scores.get(idx, 0) + (
            1 / (60 + rank + 1)
        )

    # --------------------------------
    # Metadata-based Ranking
    # --------------------------------

    for idx in combined_scores:

        chunk = chunks[idx]

        course = chunk.get("course", "").lower()
        unit = chunk.get("unit", "").lower()

        # Boost matching course
        if target_course and target_course.lower() in course:
            combined_scores[idx] += 0.05

        # Boost matching unit
        if target_unit:

            unit_number = re.search(r"\bunit\s+(\d+)\b", unit)

            if unit_number and int(unit_number.group(1)) == target_unit:
                combined_scores[idx] += 0.05

    # --------------------------------
    # Final Ranking
    # --------------------------------

    ranked_indices = sorted(
        combined_scores,
        key=combined_scores.get,
        reverse=True
    )

    # Prefer exact course + unit matches
    if target_course and target_unit:

        matching = [
            idx for idx in ranked_indices
            if target_course.lower() in chunks[idx].get("course", "").lower()
            and re.search(
                rf"\bunit\s+{target_unit}\b",
                chunks[idx].get("unit", ""),
                re.IGNORECASE
            )
        ]

        remaining = [
            idx for idx in ranked_indices
            if idx not in matching
        ]

        ranked_indices = matching + remaining

    ranked_indices = ranked_indices[:top_k]

    # --------------------------------
    # Return Results
    # --------------------------------

    results = []

    for idx in ranked_indices:

        results.append({
            "page": chunks[idx]["page"],
            "course": chunks[idx].get("course", ""),
            "unit": chunks[idx].get("unit", ""),
            "text": chunks[idx].get("original_text", chunks[idx]["text"]),
            "score": float(combined_scores[idx])
        })

    return results



# ==========================================
# TEST RETRIEVAL
# ==========================================

if __name__ == "__main__":

    query = input("\nAsk a question about your syllabus: ")

    results = retrieve(query, top_k=5)

    print("\nHybrid Search Results:\n")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print(f"Page: {result['page']}")
        print(f"Combined Score: {result['score']:.4f}")
        print(result["text"])
        print("-" * 60)
