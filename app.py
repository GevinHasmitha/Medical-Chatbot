from flask import Flask, render_template, jsonify, request
from openai import chat
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_pinecone import PineconeVectorStore

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


embeddings = download_huggingface_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(model="gpt-5-nano")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)  # This takes the retrieved documents/embeddings and "stuffs" them into the prompt
rag_chain = create_retrieval_chain(retriever, question_answer_chain) # Wraps everything in a full RAG pipeline


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form.get("msg")
    input = msg.strip()
    print(f"Received message: {input}")
    response = rag_chain.invoke({"input": msg})
    print(f"Generated response: {response}")
    return str(response["answer"])



    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)