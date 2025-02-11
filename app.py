import subprocess
import sys

def install_packages():
    """Force install missing dependencies."""
    packages = ["numpy==1.23.5", "scipy==1.8.1", "gensim==4.1.2", "Cython", "setuptools==65.5.0", "wheel", "smart_open"]
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError:
            print(f"⚠️ Failed to install {package}, skipping...")

# Try importing gensim and install if missing
try:
    from gensim.models import Word2Vec
except ImportError:
    install_packages()
    from gensim.models import Word2Vec  # Retry import



import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO


# ✅ Ensure gensim is installed
try:
    from gensim.models import Word2Vec
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "gensim"])
    from gensim.models import Word2Vec

st.title('Embedding Similarity App')

# ✅ Corrected GitHub raw URL for embeddings
GITHUB_URL = "https://raw.githubusercontent.com/AhmedUdst/mlops/main/document_embeddings.npy"

@st.cache_data
def load_embeddings(url):
    """Load document embeddings from GitHub"""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            st.success("✅ Successfully loaded embeddings from GitHub.")
            return np.load(BytesIO(response.content))
        else:
            st.error(f"❌ Failed to load embeddings. HTTP Status Code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"⚠️ Error loading embeddings: {str(e)}")
        return None

@st.cache_resource
def train_word2vec():
    """Train a simple Word2Vec model on example sentences"""
    st.info("⏳ Training Word2Vec model...")
    sample_sentences = [
        ["this", "is", "an", "example", "sentence"],
        ["word", "embedding", "models", "are", "useful"],
        ["streamlit", "is", "a", "great", "tool"],
        ["natural", "language", "processing", "is", "interesting"],
        ["word2vec", "is", "used", "for", "similarity"]
    ]
    
    model = Word2Vec(sentences=sample_sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    st.success("✅ Word2Vec model trained successfully.")
    return model

def get_sentence_embedding(sentence, model):
    """Convert user input into a vector using trained Word2Vec"""
    words = sentence.lower().split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]

    if not word_vectors:  # If no words are found in the model
        return None

    return np.mean(word_vectors, axis=0)  # Take the average of word embeddings

# ✅ Load embeddings
embeddings = load_embeddings(GITHUB_URL)

# ✅ Train Word2Vec model
word2vec_model = train_word2vec()

# ✅ Check if embeddings loaded correctly
if embeddings is not None and word2vec_model is not None:
    st.write(f"✅ Embeddings shape: {embeddings.shape}")

    # Model selection dropdown
    models = ['Model A', 'Model B', 'Model C']
    selected_model = st.selectbox('Select a model:', models)

    # User text input
    user_input = st.text_input('Enter your text:')

    # Submit button
    if st.button('Submit'):
        if user_input.strip():
            user_embedding = get_sentence_embedding(user_input, word2vec_model)

            if user_embedding is not None:
                user_embedding = user_embedding.reshape(1, -1)  # Ensure correct shape

                # Compute cosine similarity
                similarities = cosine_similarity(user_embedding, embeddings)

                # Get top-k most similar indexes
                top_k = 5
                top_k_indexes = np.argsort(similarities[0])[-top_k:][::-1]

                # Display results
                st.write('🔥 Top-k most similar indexes:', top_k_indexes)
            else:
                st.warning("⚠️ No valid words found in Word2Vec. Please try a different input.")
        else:
            st.warning("⚠️ Please enter text before submitting.")
else:
    st.error("❌ Could not load embeddings or Word2Vec model. Please check your setup.")
