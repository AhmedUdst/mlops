import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

st.title('Embedding Similarity App')

uploaded_file = st.file_uploader("Upload a numpy array of embeddings", type=["npy"])

if uploaded_file is not None:
    # Load the numpy array
    embeddings = np.load(BytesIO(uploaded_file.read()))

    # Display the shape of the embeddings
    st.write(f"Embeddings shape: {embeddings.shape}")

    # Define the list of models (for demonstration purposes)
    models = ['Model A', 'Model B', 'Model C']

    # Create a drop-down list for model selection
    selected_model = st.selectbox('Select a model:', models)

    # Create an input box for user text input
    user_input = st.text_input('Enter your text:')

    # Create a submit button
    if st.button('Submit'):
        # Placeholder for converting user input to embeddings
        # Replace this with actual model prediction logic
        user_embedding = np.random.rand(1, embeddings.shape[1])

        # Calculate cosine similarity
        similarities = cosine_similarity(user_embedding, embeddings)

        # Get the top-k most similar indexes
        top_k = 5
        top_k_indexes = np.argsort(similarities[0])[-top_k:][::-1]

        # Display the top-k most similar indexes
        st.write('Top-k most similar indexes:', top_k_indexes)
