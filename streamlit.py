import streamlit as st
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from langchain.chains import RetrievalQA  # Assuming langchain is installed
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp  # Assuming LlamaCpp is installed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # Assuming FAISS is installed
from langchain.document_loaders import PyPDFDirectoryLoader


# Load pre-trained model and tokenizer (replace with your choices)
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = HuggingFaceEmbeddings.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load or create vector store (adjust paths as needed)
try:
    vector_store = FAISS.load_local("/content/drive/MyDrive/Model/faiss_1")
except FileNotFoundError:
    # Load PDFs from a directory (replace with your data path)
    pdf_directory = "/content/sample_data/data"  # Adjust path if needed
    loader = PyPDFDirectoryLoader(pdf_directory)
    data = loader.load()

    # Process text chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(data)

    # Create vector store
    vector_store = FAISS.from_documents(text_chunks, embedding=model)

# Load or create LLM model (adjust path as needed)
try:
    llm = LlamaCpp(
        streaming=True,
        model_path="/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.75,
        top_p=1,
        verbose=True,
        n_ctx=4096
    )
except FileNotFoundError:
    st.error("LLM model not found. Please provide the path to your model.")
    sys.exit(1)

# Create the question-answering chain using Streamlit and Gradio
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2})
)

# Streamlit app layout
st.title("Question Answering App")

# Text input with Gradio
def answer_query(query):
    return qa.run(query)["result"]

iface = gr.Interface(
    fn=answer_query,
    inputs="text",
    outputs="text",
    title="Ask your question here:",
    theme="automatic"  # Optional for a more user-friendly theme
)

st.write(iface)

# Speech-to-text functionality (optional)
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice input...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"Voice Input: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I did not understand the audio.")
            return None
        except sr.RequestError:
            print("Sorry, there was a problem with the request.")
            return None

if st.button("Enable Voice Input"):
    voice_input = get_voice_input()
    if voice_input:
        answer = answer_query(voice_input)
        st.write(f"Answer: {answer}")

# Text-to-speech functionality (optional)
def speak_answer(answer):
    engine = pyttsx3.init()
    engine.say(answer)
    engine.runAndWait()

if st.button("Enable Text-to-Speech"):
    answer = answer_query(st.text_input("Enter your question:"))
    speak_answer(answer) 


