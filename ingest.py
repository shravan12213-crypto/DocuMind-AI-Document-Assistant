
import pymupdf
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import re


# ==========================================
# STEP 1: EXTRACT TEXT FROM PDF
# ==========================================

def extract_text(pdf_path):

    document = pymupdf.open(pdf_path)

    extracted_pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text", sort=True).strip()

        if text:
            extracted_pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return extracted_pages


# ==========================================
# STEP 2: CONTEXT-AWARE CHUNKING
# ==========================================

def chunk_text(pages, chunk_size=1000, overlap=200):

    chunks = []

    current_course = "General"
    current_unit = "General"

    for page in pages:

        page_number = page["page"]
        lines = page["text"].splitlines()

        buffer = []
        buffer_length = 0

        def save_chunk(content, course, unit, page_num):

            content = content.strip()

            if not content:
                return

            contextual_text = (
                f"Course: {course}\n"
                f"{unit}\n\n"
                f"{content}"
            )

            chunks.append({
                "page": page_num,
                "course": course,
                "unit": unit,
                "text": contextual_text,
                "original_text": content
            })

        for line in lines:

            clean_line = line.strip()

            if not clean_line:
                continue

            # Detect course title and code
            course_match = re.search(
                r"\b[A-Z]{2,}\d{3,4}\s*:\s*(.+)",
                clean_line
            )

            if course_match:

                current_course = (
                    clean_line.strip()
                )

            # Detect unit heading
            unit_match = re.search(
                r"Unit\s+(\d+)\s*:\s*(.+)",
                clean_line,
                re.IGNORECASE
            )

            if unit_match:

                # Save existing content before changing unit
                save_chunk(
                    "\n".join(buffer),
                    current_course,
                    current_unit,
                    page_number
                )

                buffer = []
                buffer_length = 0

                current_unit = (
                    f"Unit {unit_match.group(1)}: "
                    f"{unit_match.group(2).strip()}"
                )

            # Add line to buffer
            if buffer_length + len(clean_line) > chunk_size:

                save_chunk(
                    "\n".join(buffer),
                    current_course,
                    current_unit,
                    page_number
                )

                # Retain overlap
                previous_text = "\n".join(buffer)

                overlap_text = previous_text[-overlap:] if overlap else ""

                buffer = [overlap_text] if overlap_text else []
                buffer_length = len(overlap_text)

            buffer.append(clean_line)
            buffer_length += len(clean_line)

        # Save remaining content on this page
        save_chunk(
            "\n".join(buffer),
            current_course,
            current_unit,
            page_number
        )

    return chunks



# ==========================================
# STEP 3: GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return embeddings


# ==========================================
# STEP 4: BUILD FAISS VECTOR DATABASE
# ==========================================

def build_vector_store(chunks, embeddings):

    print("\nCreating FAISS index...")

    embeddings = np.asarray(embeddings, dtype=np.float32)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    vectorstore_path = Path("vectorstore")
    vectorstore_path.mkdir(exist_ok=True)

    faiss.write_index(
        index,
        str(vectorstore_path / "index.faiss")
    )

    with open(
        vectorstore_path / "chunks.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nFAISS index created successfully!")

    print(f"Total vectors stored: {index.ntotal}")
    print(f"Vector dimensions: {dimension}")


# ==========================================
# STEP 5: Remove repeted header
# ==========================================
def clean_text(text):

    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if "Vishwakarma Institute of Technology" in line:
            continue

        if "Computer Engineering Syllabus- AY 2026-27" in line:
            continue

        if re.match(r"^Page\s+\d+$", line, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# ==========================================
# STEP 6: EXECUTE PIPELINE
# ==========================================

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
