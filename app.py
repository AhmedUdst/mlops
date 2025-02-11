import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
import gensim.downloader as api

st.title('Embedding Similarity App')

# ✅ Corrected GitHub raw URL
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
def load_word2vec():
    """Load a pre-trained Word2Vec model"""
    try:
        st.info("⏳ Loading Word2Vec model...")
        model = api.load("word2vec-google-news-300")  # Pre-trained model
        st.success("✅ Word2Vec model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading Word2Vec model: {str(e)}")
        return None

def get_sentence_embedding(sentence, model):
    """Convert user input into a vector using Word2Vec"""
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]

    if not word_vectors:  # If no words are found in the model
        return None

    return np.mean(word_vectors, axis=0)  # Take average of word embeddings

# ✅ Load embeddings
embeddings = load_embeddings(GITHUB_URL)

# ✅ Load Word2Vec model
word2vec_model = load_word2vec()

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
