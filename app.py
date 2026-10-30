import streamlit as st
import tempfile
import hashlib
import os

from ingest import (
    extract_text,
    chunk_text,
    generate_embeddings,
    build_vector_store
)

from rag import answer_question


# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide"
)


# ---------------- SESSION STATE ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.title("📚 DocuMind")

    st.caption("AI-Powered Document Assistant")

    st.divider()

    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        current_hash = hashlib.md5(
            uploaded_file.getvalue()
        ).hexdigest()

        if st.button("Process Document", use_container_width=True):

            if current_hash != st.session_state.file_hash:

                try:
                    with st.spinner("Processing your document..."):

                        # Save uploaded PDF temporarily
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as temp_file:

                            temp_file.write(uploaded_file.getvalue())
                            pdf_path = temp_file.name

                        # Extract text
                        pages = extract_text(pdf_path)

                        # Generate chunks
                        chunks = chunk_text(pages)

                        # Generate embeddings
                        embeddings = generate_embeddings(chunks)

                        # Build FAISS vector database
                        build_vector_store(chunks, embeddings)

                        # Clean up temporary file
                        os.remove(pdf_path)

                        st.session_state.file_hash = current_hash
                        st.session_state.document_ready = True
                        st.session_state.messages = []

                    st.success("Document processed successfully!")

                except Exception as e:
                    st.error(f"Document processing failed: {e}")

            else:
                st.session_state.document_ready = True
                st.info("This document is already processed.")

    st.divider()

    if st.session_state.document_ready:
        st.success("Document Ready")
    else:
        st.info("Upload and process a PDF to begin.")

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- MAIN CHAT INTERFACE ----------------

st.title("📚 DocuMind")
st.subheader("Your AI-powered Document Assistant")

st.write(
    "Upload a PDF and ask questions. "
    "DocuMind retrieves relevant information and generates "
    "context-aware answers using Gemini."
)

st.divider()


# ---------------- WELCOME MESSAGE ----------------

if not st.session_state.messages:

    st.markdown("### Welcome to DocuMind! 👋")

    st.write("You can ask questions such as:")

    st.markdown(
        """
        - What topics are covered in recurrence relations?
        - Explain the Tower of Hanoi.
        - What is covered in Graph Theory?
        - Summarise the course objectives.
        """
    )


# ---------------- DISPLAY CHAT HISTORY ----------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------- USER INPUT ----------------

question = st.chat_input("Ask something about your document...")

if question:

    if not st.session_state.document_ready:

        st.warning("Please upload and process a PDF first.")

    else:

        # Display user question
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Searching your document..."):

                try:
                    answer = answer_question(question)

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Unable to generate answer: {e}")