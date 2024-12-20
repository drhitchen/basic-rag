# RAG Workflow Project

## Description
This project implements a basic Retrieval-Augmented Generation (RAG) workflow using LangChain and OpenAI's GPT models. It leverages a local ChromaDB document store and Gradio for user interaction.

## Requirements
- Python 3.8 or newer

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

### 5. Add Local Documents
Place your documents (e.g., PDFs) in the designated `data/` folder. Or, to test with the sample keto shopping list from Dr. Gundry, copy `UnlockTheKetoCodeShoppingList.pdf` to the `data/` folder. Here are some prompts to try with this file.

#### Sample Prompts:

- **Create a numbered list of ALL items on Dr. Gundry's "No" list.**
- **Create a numbered list of ALL items on Dr. Gundry's "Yes" list.**

- **Create a numbered list of ALL items on Dr. Gundry's "No" list. Give me the full list which should contain nearly 100 items.**
- **Create a numbered list of ALL items on Dr. Gundry's "Yes" list. Give me the full list which should contain over 100 items.**

- **You are a specialized assistant. Retrieve and respond only using the uploaded document content. Do not speculate or rely on general knowledge. Cite the document for all responses. Create a numbered list of ALL items on Dr. Gundry's "No" list. Give me the full list which should contain nearly 100 items.**
- **You are a specialized assistant. Retrieve and respond only using the uploaded document content. Do not speculate or rely on general knowledge. Cite the document for all responses. Create a numbered list of ALL items on Dr. Gundry's "Yes" list. Give me the full list which should contain over 100 items.**

### 6. Run the Application
Execute the script to start the Gradio interface:
```bash
python rag-app01.py
```

### 7. Open and interact with UI in web browser
Browse to http://127.0.0.1:1337:
```bash
open http://127.0.0.1:1337
```

## Features
- **Document Retrieval**: Uses LangChain to retrieve context from local documents.
- **Question Answering**: Provides responses using OpenAI's GPT model.
- **Interactive Interface**: Built with Gradio for user-friendly interaction.

## **License**

This project is licensed under the [MIT License](LICENSE).

## **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request.
