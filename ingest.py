import pymupdf
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


# STEP 1: Extract text from PDF

def extract_text(pdf_path):

    document = pymupdf.open(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        if text:
            extracted_pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return extracted_pages


# STEP 2: Divide text into smaller chunks

def chunk_text(pages, chunk_size=1000, overlap=200):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "page": page_number,
                    "text": chunk
                })

            start += chunk_size - overlap

    return chunks


# STEP 3: Generate embeddings

def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    print("Generating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings

# STEP 4: Create and save FAISS vector database

def build_vector_store(chunks, embeddings):

    print("\nCreating FAISS index...")

    # Convert embeddings into float32
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings)

    # Get embedding dimensions
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to index
    index.add(embeddings)

    # Create vectorstore directory
    vectorstore_path = Path("vectorstore")
    vectorstore_path.mkdir(exist_ok=True)

    # Save FAISS index
    faiss.write_index(
        index,
        str(vectorstore_path / "index.faiss")
    )

    # Save original chunks and metadata
    with open(vectorstore_path / "chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

    print("FAISS index created successfully!")

    print(f"Total vectors stored: {index.ntotal}")

    print(f"Vector dimensions: {dimension}")


# STEP 4: Execute the pipeline

if __name__ == "__main__":

    pdf_path = Path("data/syllabus.pdf")

    if not pdf_path.exists():

        print("PDF file not found!")

    else:

        pages = extract_text(pdf_path)

        print(f"Total pages extracted: {len(pages)}")

        chunks = chunk_text(pages)

        

        print(f"Total chunks generated: {len(chunks)}")

        embeddings = generate_embeddings(chunks)
        
        build_vector_store(chunks, embeddings)

        print("\nEmbedding generation completed!")

        print(f"Embedding shape: {embeddings.shape}")

        print(f"First embedding:\n{embeddings[0]}")