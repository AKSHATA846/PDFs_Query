import streamlit as st
import requests
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

# Rest of your app code...
