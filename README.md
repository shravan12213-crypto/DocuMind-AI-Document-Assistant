# DocuRAG – Retrieval-Augmented Document Assistant

DocuRAG is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content. Using Sentence Transformers, FAISS and Gemini, it retrieves relevant information and generates context-aware answers with source citations.

## 🚀 Features

* **PDF Upload:** Upload syllabus, research papers, academic documents and other PDFs.
* **Text Extraction:** Extract text from PDF documents using PyMuPDF.
* **Text Chunking:** Divide documents into smaller, manageable chunks.
* **Semantic Search:** Retrieve relevant information using embeddings and FAISS.
* **AI-Powered Responses:** Generate answers using Google's Gemini API.
* **Source Citations:** Display document names and page numbers for retrieved information.
* **Hallucination Control:** Indicate when information cannot be found in the uploaded documents.

## 🛠️ Tech Stack

| Technology            | Purpose                              |
| --------------------- | ------------------------------------ |
| Python                | Core programming language            |
| Streamlit             | User interface                       |
| PyMuPDF               | PDF text extraction                  |
| Sentence Transformers | Text embeddings                      |
| FAISS                 | Vector storage and similarity search |
| Google Gemini API     | Response generation                  |
| NumPy                 | Numerical operations                 |

## ⚙️ RAG Architecture

DocuRAG follows a standard Retrieval-Augmented Generation pipeline.

1. **Document Ingestion:** Extract text from uploaded PDF documents.
2. **Chunking:** Split extracted text into smaller chunks with overlapping content.
3. **Embedding Generation:** Convert text chunks into numerical vector representations using Sentence Transformers.
4. **Vector Storage:** Store embeddings in a FAISS vector index.
5. **Retrieval:** Convert user queries into embeddings and retrieve the most relevant chunks using similarity search.
6. **Augmentation:** Combine retrieved context with the user's question.
7. **Generation:** Send the context to Gemini to generate a relevant answer.
8. **Citation:** Display the answer along with its source document and page number.

## 📁 Project Structure

```text
DocuRAG/
│
├── app.py
├── ingest.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   └── syllabus.pdf
│
├── vectorstore/
│
└── README.md
```

## 💻 Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd DocuRAG
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Gemini API Key

Create a `.env` file in the project root directory.

```env
GEMINI_API_KEY=your_api_key_here
```

Get your API key from Google AI Studio.

### 5. Run the Application

```bash
streamlit run app.py
```

## 🧪 Evaluation

The RAG chatbot will be evaluated using at least 20 questions across three categories:

| Test Category            | Number of Questions |
| ------------------------ | ------------------: |
| Direct factual questions |                   8 |
| Conceptual questions     |                   7 |
| Unanswerable questions   |                   5 |
| **Total**                |              **20** |

Evaluation will consider:

* Answer correctness
* Retrieval relevance
* Source citation accuracy
* Handling of unanswerable questions

## 🎯 Use Cases

* Understanding academic syllabi.
* Searching college academic policies.
* Asking questions about research papers.
* Extracting information from lengthy PDF documents.
* Quickly finding specific information in documents.

## 🔮 Future Improvements

* Support for multiple PDF documents.
* Chat history and conversational memory.
* Improved retrieval using reranking.
* Support for additional document formats.
* More comprehensive retrieval evaluation.

## 👨‍💻 Author

**Shravan Dilip More**

B.Tech Computer Engineering
Vishwakarma Institute of Technology, Pune

---

*Built as a practical implementation of Retrieval-Augmented Generation (RAG).*

