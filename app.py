import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_chunk, build_vectorstore, get_retriever, get_answer

st.title("RAG Explorer")
st.caption("Compare retrieval strategies on your own PDF")

#Sidebar
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    retriever_type = st.selectbox(
        "Select Retriever",
        ["Similarity", "MMR", "Parent Document", "Multi Query"]
    )
    st.info("""
    **Similarity** — baseline, top k chunks\n
    **MMR** — reduces redundancy in retrieved chunks\n
    **Parent Document** — retrieves small chunks, returns large parent chunks to LLM\n
    **Multi Query** — generates multiple query rephrasings to improve coverage
    """)

# Main area
if uploaded_file is not None:
    # Only process if new file uploaded
    if "processed" not in st.session_state or st.session_state.processed != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Loading and chunking PDF..."):
            documents, chunks = load_and_chunk(tmp_path)
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            st.session_state.processed = uploaded_file.name

        with st.spinner("Building vector store..."):
            vectorstore, embeddings = build_vectorstore(st.session_state.chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.embeddings = embeddings

        st.success(f"Loaded {len(st.session_state.documents)} pages, {len(st.session_state.chunks)} chunks")

    question = st.text_input("Ask a question about your PDF")

    if question:
        with st.spinner(f"Retrieving with {retriever_type}..."):
            retriever = get_retriever(
                retriever_type,
                st.session_state.vectorstore,
                st.session_state.documents,
                st.session_state.embeddings
            )
            answer, source_docs = get_answer(question, retriever)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        for i, doc in enumerate(source_docs):
            with st.expander(f"Chunk {i+1} — Page {doc.metadata.get('page', 'N/A')}"):
                st.write(doc.page_content)
else:
    st.info("Upload a PDF from the sidebar to get started")