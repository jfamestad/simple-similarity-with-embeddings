import streamlit as st
from sentence_transformers import SentenceTransformer, util

if not "embeddings_model" in st.session_state:
    st.session_state.embeddings_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

if not "text_inputs" in st.session_state:
    st.session_state.text_inputs = None

if not "text_query" in st.session_state:
    st.session_state.text_query = ""

if not "chunked_text_input_scores" in st.session_state:
    st.session_state.chunked_text_input_scores = {}
def update_sorted():
    chunked_text_inputs = st.session_state.text_inputs.split("\n")
    chunked_text_inputs = [i for i in chunked_text_inputs if not i == '']
    print(f"Chunks: {str(chunked_text_inputs)}")

    text_query_embedding = st.session_state.embeddings_model.encode(st.session_state.text_query)

    st.session_state.chunked_text_input_scores = []
    for chunk in chunked_text_inputs:
        embedding = st.session_state.embeddings_model.encode(chunk)
        score = float(util.dot_score(text_query_embedding, embedding)[0][0])
        st.session_state.chunked_text_input_scores.append({
            "text": chunk,
            "embedding": embedding,
            "score": score,
            "formatted_output": f"{score} {chunk}"
        })

st.title("Similarity Explorer")

with st.sidebar:
    "Configure Embeddings Model"

st.session_state.text_inputs = st.text_area(
    label="Inputs - Will be chunked by newlines",
    on_change=update_sorted
)

st.session_state.text_query = st.text_input(
    label="Query text",
    on_change=update_sorted
)

print(f"Sorted inputs")
for line in sorted(st.session_state.chunked_text_input_scores, key=lambda x: x['score'], reverse=True):
    st.write(line["formatted_output"])