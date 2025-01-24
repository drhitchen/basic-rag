
# RAG Workflow Project

## Description
This project implements a basic Retrieval-Augmented Generation (RAG) workflow using LangChain, OpenAI's GPT models, and a local ChromaDB document store. It provides multiple examples to demonstrate the evolution of the RAG system, starting with a basic implementation and adding features incrementally.

## Requirements
- Python 3.8 or newer

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/drhitchen/basic-rag.git
cd basic-rag
```

### 2. Create and Activate a Virtual Environment
#### Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```
#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root and add the following line with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key
```

---

## Learning Path: Examples in Order

This section provides a walkthrough of the RAG examples in the repository, highlighting what was added or changed in each iteration.

### 1. **Basic Gradio Implementation (rag-app01.py)**

This is the foundational RAG example:
- Implements document loading and text splitting using LangChain.
- Creates embeddings using OpenAI's `text-embedding-ada-002` model.
- Uses ChromaDB for storing embeddings.
- Provides a Gradio interface for asking questions about documents.

#### Features:
- Processes all PDFs in a specified `data/` folder. Place your documents (e.g., PDFs) in the `data/` folder to make them accessible for processing.
- Starts with a fresh ChromaDB instance every time.
- Returns answers to questions using OpenAI's GPT models.

**To Run:**
```bash
python rag-app01.py
```

**Access:**
Open your browser to [http://127.0.0.1:1337](http://127.0.0.1:1337) to interact with the interface.

---

### 2. **Enhanced Streamlit Implementation (rag-app02_streamlit.py)**

This version transitions to a **Streamlit** interface and introduces:
- Persistent ChromaDB storage.
- The ability to process individual PDFs rather than processing all files in a folder.
- Improved embeddings generation using OpenAI's embedding models.

#### Changes/Additions:
- Replaces the batch document processing approach with individual file uploads.
- Uses persistent storage for ChromaDB to avoid regenerating embeddings for already processed files.
- Streamlit interface allows for easier interaction.

**To Run:**
```bash
streamlit run rag-app02_streamlit.py
```

---

### 3. **Enhanced Gradio Implementation (rag-app02_gradio.py)**

This version brings the improvements from `rag-app02_streamlit.py` back to the Gradio interface:
- Adds persistent ChromaDB storage.
- Allows incremental processing of PDFs through the Gradio interface.
- Enables more interactive and user-friendly document processing.

#### Changes/Additions:
- Introduces persistent storage.
- Processes PDFs individually and updates the database.

**To Run:**
```bash
python rag-app02_gradio.py
```

---

### 4. **Streamlit with Full Control (rag-app03_streamlit.py)**

This version builds upon the Streamlit interface to add full database management features:
- Allows purging/resetting the ChromaDB database directly from the interface.
- Displays warnings if a PDF has already been processed.
- Improves query handling.

#### Changes/Additions:
- Adds a "Purge Database" button to reset ChromaDB.
- Warns users if a file is already processed.
- More robust error handling and improved user feedback.

**To Run:**
```bash
streamlit run rag-app03_streamlit.py
```

---

### 5. **Gradio with Full Control (rag-app03_gradio.py)**

This final version combines all previous improvements into the Gradio interface:
- Adds database purge/reset functionality.
- Allows incremental PDF processing.
- Lets users query the system without re-uploading documents.

#### Changes/Additions:
- "Process PDF" button for incremental processing.
- "Ask Question" button for querying existing data.
- "Purge Database" button for resetting the ChromaDB collection.

**To Run:**
```bash
python rag-app03_gradio.py
```

---

## Features Across Applications

### **Document Retrieval**
- Uses LangChain and OpenAI Embeddings to retrieve context from uploaded PDFs.

### **Question Answering**
- Provides context-aware answers using OpenAI's GPT models.

### **Interactive Interfaces**
- Supports both Gradio and Streamlit for flexible user interaction.

---

## Sample Prompts
Use these prompts for the included `UnlockTheKetoCodeShoppingList.pdf`:

1. **"Create a numbered list of ALL items on Dr. Gundry's 'No' list."**
2. **"Create a numbered list of ALL items on Dr. Gundry's 'Yes' list."**
3. **"Retrieve and respond only using the uploaded document content. Do not speculate or rely on general knowledge. Cite the document for all responses."**

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request.
