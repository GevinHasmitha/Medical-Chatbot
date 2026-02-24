

from src.helper import download_embeddings, filter_to_minimal_docs, load_pdf_files, text_split
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

extracted_data = load_pdf_files("data/")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_split(minimal_docs)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(
    api_key=pinecone_api_key,
)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(  # This tells Pinecone to create a new index in their cloud infrastructure.
        name=index_name,
        dimension=384, 
        metric="cosine", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # Serverless configuration
    )
index = pc.Index(index_name)  # This creates a Python client object that points to your index.

docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)