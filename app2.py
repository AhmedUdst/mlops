import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

st.title('Embedding Similarity App')

# ✅ Corrected GitHub raw URL
GITHUB_URL = "https://raw.githubusercontent.com/AhmedUdst/mlops/main/document_embeddings.npy"

@st.cache_data
def load_embeddings(url):
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

# ✅ Load embeddings
embeddings = load_embeddings(GITHUB_URL)

# ✅ Check if embeddings loaded correctly
if embeddings is not None:
    st.write(f"✅ Embeddings shape: {embeddings.shape}")

    # Model selection dropdown
    models = ['Model A', 'Model B', 'Model C']
    selected_model = st.selectbox('Select a model:', models)

    # User text input
    user_input = st.text_input('Enter your text:')

    # Submit button
    if st.button('Submit'):
        if user_input.strip():
            user_embedding = np.random.rand(1, embeddings.shape[1])  # Match embedding dimension

            # Compute cosine similarity
            similarities = cosine_similarity(user_embedding, embeddings)

            # Get top-k most similar indexes
            top_k = 5
            top_k_indexes = np.argsort(similarities[0])[-top_k:][::-1]

            # Display results
            st.write('🔥 Top-k most similar indexes:', top_k_indexes)
        else:
            st.warning("⚠️ Please enter text before submitting.")
else:
    st.error("❌ Could not load embeddings. Please check the GitHub URL or file permissions.")
