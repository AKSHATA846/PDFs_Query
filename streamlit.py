import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name=r"C:\Users\SLNX1\Downloads\mistral-7b-instruct-v0.1.Q4_K_M.gguf")

# Function to read PDFs and extract text
def load_pdfs_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with open(pdf_path, "rb") as file:
                reader = PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                texts.append(text)
    return texts

# Function to create embeddings and store them in FAISS
def create_faiss_index(texts):
    return FAISS.from_texts(texts, embeddings)

# Path to your PDFs folder
pdf_folder_path = r"C:\Users\SLNX1\Documents\Project_Files"  # Update this with your folder path

# Load PDFs and create FAISS index in the background
texts = load_pdfs_from_folder(pdf_folder_path)
faiss_index = create_faiss_index(texts)

# Streamlit UI
st.title("PDF Question Answering App")
st.write("Ask questions related to the content of the PDFs.")

# User input for question
user_question = st.text_input("Enter your question:")

if user_question:
    # Search for the most relevant text in the PDFs
    docs = faiss_index.similarity_search(user_question, k=3)
    
    # Use HuggingFaceHub LLM to answer the question based on the retrieved documents
    llm = HuggingFaceHub(repo_id="gemini_model_repo_id", model_kwargs={"api_key": "AIzaSyDAXHJaUBYHzw7L1CS5Sj7n0htWTEU4fsA"})  # Replace with your Gemini model repo ID and API key
    chain = load_qa_chain(llm, chain_type="stuff")
    
    # Get the answer
    answer = chain.run(input_documents=docs, question=user_question)
    
    # Display the answer
    st.write("### Answer:")
    st.write(answer)

    # Optionally, display the most relevant documents
    st.write("### Relevant Document Passages:")
    for doc in docs:
        st.write(doc.page_content)
