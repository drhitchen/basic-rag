import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from chromadb import Client
from chromadb.config import Settings

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")

client = OpenAI(api_key=api_key)

assistant_config = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

client_chroma = Client(Settings(persist_directory=".chromadb"))
collection = client_chroma.create_collection("pdf_collection")

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "".join([page.extract_text() for page in reader.pages])

def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(pdf_text)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]
    for i, chunk in enumerate(chunks):
        collection.add(ids=[f"chunk_{i}"], documents=[chunk], embeddings=[chunk_embeddings[i]])

def query_system(query):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    query_embedding = embeddings.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    relevant_chunks = [doc for doc in results['documents'][0]]
    context = "\n".join(relevant_chunks)
    response = client.chat.completions.create(
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
    return response.choices[0].message.content

# Gradio interface
def handle_query(pdf, question):
    process_pdf(pdf.name)
    return query_system(question)

gr.Interface(
    fn=handle_query,
    inputs=["file", "text"],
    outputs="text",
    title="PDF Question Answering System",
    description="Upload a PDF and ask questions about its content."
).launch()
