import gradio as gr
import openai
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from chromadb import Client
from chromadb.config import Settings
import hashlib

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")

assistant_config = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

client_chroma = Client(Settings(persist_directory=".chromadb"))
collection = client_chroma.get_or_create_collection("pdf_collection")


def hash_pdf_content(pdf_content):
    """Generate a SHA-256 hash of the PDF content."""
    return hashlib.sha256(pdf_content.encode('utf-8')).hexdigest()


def extract_text_from_pdf(uploaded_file):
    """Extract text from a PDF using an uploaded file object."""
    reader = PdfReader(uploaded_file)
    return "".join([page.extract_text() for page in reader.pages])


def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)


def process_pdf(uploaded_file):
    """
    Process the uploaded PDF file, generate embeddings, and add them to the collection.
    """
    pdf_text = extract_text_from_pdf(uploaded_file)
    pdf_hash = hash_pdf_content(pdf_text)
    file_name = os.path.basename(uploaded_file.name)

    # Check if content is already in the collection
    existing_documents = collection.get(where={"id": f"{pdf_hash}_chunk_0"})  # Check first chunk ID
    if existing_documents["ids"]:
        return f"Content of '{file_name}' is already processed. Skipping."

    chunks = split_text_into_chunks(pdf_text)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]

    # Add chunks and embeddings to the collection
    for i, chunk in enumerate(chunks):
        document_with_metadata = f"File Name: {file_name}\nChunk: {i}\nContent:\n{chunk}"
        collection.add(
            ids=[f"{pdf_hash}_chunk_{i}"],  # Use hash to uniquely identify chunks
            documents=[document_with_metadata],
            embeddings=[chunk_embeddings[i]]
        )
    return f"Processed and stored embeddings for '{file_name}'."


def query_system(query):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    query_embedding = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    relevant_chunks = [doc for doc in results['documents'][0]]
    context = "\n".join(relevant_chunks)
    
    # Call the OpenAI chat completion
    response = openai.chat.completions.create(
        model=assistant_config["model"],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ],
        temperature=assistant_config["temperature"],
        max_tokens=assistant_config["max_tokens"],
        top_p=assistant_config["top_p"],
        frequency_penalty=assistant_config["frequency_penalty"],
        presence_penalty=assistant_config["presence_penalty"]
    )
    
    # Extract the content properly from the response object
    return response.choices[0].message.content


def purge_database():
    """
    Purge the ChromaDB collection to remove all stored embeddings and reset the database.
    """
    collection.delete(where={"*": "*"})  # Deletes all documents in the collection
    return "Database has been purged successfully!"


# Gradio Interface
def handle_process(pdf):
    return process_pdf(pdf)


def handle_query(question):
    return query_system(question)


def handle_purge():
    return purge_database()


with gr.Blocks() as interface:
    gr.Markdown("### PDF Question Answering System")
    gr.Markdown("Upload a PDF, process it, and ask questions about its content. You can also purge the database.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_output = gr.Textbox(label="Processing Output", interactive=False)

    process_button = gr.Button("Process PDF")
    process_button.click(
        fn=handle_process,
        inputs=[pdf_input],
        outputs=[process_output]
    )

    with gr.Row():
        question_input = gr.Textbox(label="Enter your question")
        answer_output = gr.Textbox(label="Answer", interactive=False)

    question_button = gr.Button("Ask Question")
    question_button.click(
        fn=handle_query,
        inputs=[question_input],
        outputs=[answer_output]
    )

    purge_button = gr.Button("Purge Database")
    purge_button.click(
        fn=handle_purge,
        inputs=[],
        outputs=[process_output]
    )

interface.launch()
