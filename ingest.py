import pymupdf
from pathlib import Path
from sentence_transformers import SentenceTransformer


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

        print("\nEmbedding generation completed!")

        print(f"Embedding shape: {embeddings.shape}")

        print(f"First embedding:\n{embeddings[0]}")