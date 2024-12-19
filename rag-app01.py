import os
import openai
import glob
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import gradio as gr
import getpass

# Function to get API key
def get_openai_api_key():
    # 1. Check environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 2. Check .env file in the current working directory
    load_dotenv(os.path.expanduser("./.env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 3. Check .env file in ~/.config/fabric/
    load_dotenv(os.path.expanduser("~/.config/fabric/.env"))
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key

    # 4. Prompt the user to enter the API key
    print("API key not found in the specified locations.")
    api_key = getpass.getpass("Please enter your OpenAI API key (* input masked): ")
    return api_key

# Debug mode toggle
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Get the API key
openai.api_key = get_openai_api_key()

# Step 1: Load and Process Documents
def load_documents(directory="./data"):
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    documents = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        docs = loader.load()
        documents.extend(docs)
    return documents

print("Loading and splitting documents...")
documents = load_documents()

# Increase chunk size and overlap to keep related context together
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print("Creating embeddings and vector store...")

# Remove any existing vector store directory to ensure a fresh start
vectorstore_dir = "./chroma_db"
if os.path.exists(vectorstore_dir):
    shutil.rmtree(vectorstore_dir)

# Use the model param with OpenAIEmbeddings from langchain_openai
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
db = Chroma.from_documents(docs, embeddings, collection_name="my_pdfs", persist_directory=vectorstore_dir)

# Increase k to retrieve more chunks
retriever = db.as_retriever(search_kwargs={"k":10})

# Step 3: Create a RetrievalQA chain with chain_type="stuff"
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0),
    chain_type="stuff",
    retriever=retriever
)

# Step 4: Define a query function for Gradio
def chat_with_pdfs(query):
    if not query.strip():
        return "Please enter a question."
    response = qa.invoke(query)  # This returns a dict with keys "query" and "result"
    answer = response["result"]  # Extract the textual result
    return answer

# Step 5: Build a simple Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chat with your local PDFs")
    gr.Markdown("Type a question and hit 'Submit' to query your documents.")

    query = gr.Textbox(lines=2, placeholder="Ask something about your internal policies...")
    submit = gr.Button("Submit")
    output = gr.Textbox(label="Answer")

    submit.click(fn=chat_with_pdfs, inputs=query, outputs=output)

demo.launch(server_name="127.0.0.1", server_port=1337)
