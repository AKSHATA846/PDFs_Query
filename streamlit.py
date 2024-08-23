import streamlit as st
import gradio as gr
import requests
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
import speech_recognition as sr
import pyttsx3

# Load pre-trained SentenceTransformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

# Define the GitHub URLs for the FAISS files
github_base_url = "https://github.com/PDFs_Query/Documents/"
faiss_index_url = github_base_url + "Documents/index.faiss/index%20(1).faiss"
faiss_index_pkl_url = github_base_url + "Documents/index.faiss/index.pkl"

# Define local paths where the files will be saved
local_index_path = "Documents/index.faiss/index (1).faiss"
local_index_pkl_path = "Documents/index.faiss/index.pkl"

# Function to download files from GitHub
def download_file(url, local_path):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        f.write(response.content)

# Download the FAISS files from GitHub if they don't exist locally
if not os.path.exists(local_index_path) or not os.path.exists(local_index_pkl_path):
    st.warning("Downloading FAISS index files from GitHub...")
    download_file(faiss_index_url, local_index_path)
    download_file(faiss_index_pkl_url, local_index_pkl_path)

# Load or create FAISS vector store
try:
    vector_store = FAISS.load_local(local_index_path, embedding_model, allow_dangerous_deserialization=True)
except FileNotFoundError:
    st.error("FAISS index files could not be found or loaded.")
    st.stop()

# Load or create LLM model (adjust path as needed)
llm_model_path = "/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

try:
    llm = LlamaCpp(
        model_path=llm_model_path,
        streaming=True,
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )
except FileNotFoundError:
    st.error("LLM model not found. Please provide the correct path to your model.")
    st.stop()

# Create the question-answering chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={"k": 2})
)

# Streamlit app layout
st.title("Question Answering App")

# Gradio Interface for question answering
def answer_query(query):
    return qa_chain.run(query)["result"]

iface = gr.Interface(
    fn=answer_query,
    inputs="text",
    outputs="text",
    title="Ask your question here:",
    theme="automatic"  # Optional for a more user-friendly theme
)

st.write("### Ask a question:")
iface.launch(share=False, inline=True)

# Optional: Voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for voice input...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.write(f"Voice Input: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I did not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Sorry, there was a problem with the request.")
            return None

if st.button("Enable Voice Input"):
    voice_input = get_voice_input()
    if voice_input:
        answer = answer_query(voice_input)
        st.write(f"Answer: {answer}")

# Optional: Text-to-speech functionality
def speak_answer(answer):
    engine = pyttsx3.init()
    engine.say(answer)
    engine.runAndWait()

if st.button("Enable Text-to-Speech"):
    question = st.text_input("Enter your question:")
    if question:
        answer = answer_query(question)
        speak_answer(answer)
        st.write(f"Answer: {answer}")
