import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Streamlit app title and description
st.title("Chat with Multiple PDFs")
st.write("Upload PDF files and interact with them using embeddings.")

# Upload PDF files
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the file name
        st.write(f"Processing file: {uploaded_file.name}")
        
        # Extract text from PDF
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        
        st.write("Extracted text from PDF:")
        st.write(text)
        
        # Generate embeddings
        st.write("Generating embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        embedding = embeddings.embed(text)
        
        st.write("Embedding generated:")
        st.write(embedding)

    st.success("Processing complete!")

