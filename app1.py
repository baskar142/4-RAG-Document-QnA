import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please check your .env file.")

# Initialize embeddings and language model
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    st.error(f"Error initializing HuggingFace embeddings: {e}")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        if not os.path.exists("research_papers"):
            st.error("The 'research_papers' directory does not exist.")
            return
        st.session_state.embeddings = embeddings
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Database is ready.")

# Streamlit app layout
st.title("RAG Document Q&A With Groq And Llama3")

# User input for query
user_prompt = st.text_input("Enter your query from the research paper")

# Button to initialize vector embeddings
if st.button("Document Embedding"):
    create_vector_embedding()

# Handle query and retrieval
if user_prompt and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.time()
    response = retrieval_chain.invoke({'input': user_prompt})
    st.write(f"Response time: {time.time() - start:.2f} seconds")

    if response and 'answer' in response:
        st.write(response['answer'])

        # Display similar documents in an expander
        with st.expander("Document Similarity Search"):
            for doc in response.get('context', []):
                st.write(doc.page_content)
                st.write('------------------------')
    else:
        st.error("No answer could be generated. Please try again.")
else:
    st.info("Please provide a query and ensure the document embedding is initialized.")
