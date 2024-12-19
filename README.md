
# RAG Workflow Project

## Description
This project implements a basic Retrieval-Augmented Generation (RAG) workflow using LangChain and OpenAI's GPT models. It leverages a local document store and Gradio for user interaction.

## Requirements
- Python 3.8 or newer

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create and Activate a Virtual Environment
#### Linux/macOS:
```bash
python3 -m venv venv
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
Place your documents (e.g., PDFs) in the designated `data/` folder. These will be indexed and used by the RAG workflow.

### 6. Run the Application
Execute the script to start the Gradio interface:
```bash
python rag-app01.py
```

## Features
- **Document Retrieval**: Uses LangChain to retrieve context from local documents.
- **Question Answering**: Provides responses using OpenAI's GPT model.
- **Interactive Interface**: Built with Gradio for user-friendly interaction.

## **License**

This project is licensed under the [MIT License](LICENSE).

## **Contributing**

Contributions are welcome! Please fork the repository and submit a pull request.
