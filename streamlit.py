import streamlit as st
import gradio as gr
import requests
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
#from llama_cpp import LlamaCpp
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import speech_recognition as sr
# Import the correct module for LLaMA
from langchain_community.llms import LlamaCpp
#from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
import pyttsx3
import os

# Define the GitHub URLs for the FAISS files
github_base_url = "https://raw.githubusercontent.com/AKSHATA846/PDFs_Query/main/Documents/"
faiss_index_url = github_base_url + "index.faiss"
faiss_index_pkl_url = github_base_url + "index.pkl"

# Define local paths where the files will be saved temporarily
local_index_path = "index.faiss"
local_index_pkl_path = "index.pkl"

# Function to download files from GitHub
def download_file(url, local_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        with open(local_path, "wb") as f:
            f.write(response.content)
        st.success(f"Downloaded file from {url} to {local_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading file from {url}: {e}")

# Download the FAISS files from GitHub
st.info("Downloading FAISS index files from GitHub...")
download_file(faiss_index_url, local_index_path)
download_file(faiss_index_pkl_url, local_index_pkl_path)

# Check if files are downloaded
if not os.path.exists(local_index_path) or not os.path.exists(local_index_pkl_path):
    st.error("One or both files were not downloaded successfully.")
    st.stop()

# Check file sizes (optional, but helps in verifying correct download)
if os.path.getsize(local_index_path) == 0 or os.path.getsize(local_index_pkl_path) == 0:
    st.error("One or both files are empty.")
    st.stop()

# Load or create FAISS vector store
try:
    # Load FAISS index
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(local_index_path, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading FAISS vector store: {e}")
    st.stop()

# Load or create LLM model
llm_model_url = "https://github.com/AKSHATA846/PDFs_Query/raw/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
llm_model_path = "/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# Download the LLM model file
def download_llm_model(url, local_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        with open(local_path, "wb") as f:
            f.write(response.content)
        st.success(f"Downloaded model from {url} to {local_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model from {url}: {e}")

st.info("Downloading LLM model file...")
download_llm_model(llm_model_url, llm_model_path)

# Load LLM model
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
