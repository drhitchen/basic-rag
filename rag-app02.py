from openai import OpenAI
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings  # Updated import
from chromadb import Client
from chromadb.config import Settings

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set OPENAI_API_KEY in the .env file.")

# Instantiate the OpenAI client
client = OpenAI(api_key=api_key)

# Assistant configuration
assistant_config = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
}

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Split text into chunks
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Initialize vector database
client_chroma = Client(Settings(persist_directory=".chromadb"))  # Persistent storage
collection = client_chroma.create_collection("pdf_collection")

# Process PDF and store embeddings
def process_pdf(pdf_path):
    pdf_text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(pdf_text)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    chunk_embeddings = [embeddings.embed_query(chunk) for chunk in chunks]

    # Add chunks and embeddings to the collection
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk],
            embeddings=[chunk_embeddings[i]]
        )
    return chunks

# Query the system
def query_system(query):
    embeddings = OpenAIEmbeddings(api_key=api_key)
    query_embedding = embeddings.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # Retrieve top 5 relevant chunks
    )
    relevant_chunks = [doc for doc in results['documents'][0]]

    # Combine relevant chunks and pass to OpenAI's GPT
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

# Example Usage
if __name__ == "__main__":
    # Process a PDF file
    pdf_path = "example.pdf"
    process_pdf(pdf_path)

    # Ask a question about the PDF
    question = "What is the main topic of the PDF?"
    answer = query_system(question)
    print("Answer:", answer)
