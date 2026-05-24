import streamlit as st
import tempfile
import os
from rag_pipeline import execute_chain

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="🏦",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────
st.title("🏦 FinSight AI")
st.markdown("*Financial Document Q&A — Powered by RAG + GPT-4o mini*")
st.divider()

# ── Session State (conversation history) ─────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

# ── Sidebar — PDF Upload ──────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a financial PDF",
        type=["pdf"],
        help="Annual reports, earnings transcripts, RBI policy docs etc."
    )

    if uploaded_file is not None:
        if not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):
                # save uploaded file to temp location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    st.session_state.pdf_path = tmp.name
                st.session_state.pdf_processed = True
                st.session_state.messages = []  # reset chat on new PDF

        st.success(f"✅ {uploaded_file.name} ready!")
        st.caption("Ask any question about this document below.")

    if st.session_state.pdf_processed:
        if st.button("🗑️ Clear & Upload New PDF"):
            st.session_state.pdf_processed = False
            st.session_state.pdf_path = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown("*How it works:*")
    st.markdown("1. Upload a financial PDF")
    st.markdown("2. Ask questions in plain English")
    st.markdown("3. Get answers with page citations")

# ── Main — Chat Interface ─────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.info("👈 Upload a financial PDF from the sidebar to get started.")
    st.markdown("### Example documents to try:")
    st.markdown("- Infosys / TCS Annual Report")
    st.markdown("- RBI Monetary Policy Statement")
    st.markdown("- Any company earnings call transcript")

else:
    # display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📄 View Source Chunks"):
                    for i, chunk in enumerate(message["sources"]):
                        st.caption(f"*Source {i+1}:* {chunk}")

    # chat input
    question = st.chat_input("Ask a question about the document...")

    if question:
        # show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # get answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                try:
                    response = execute_chain(
                        st.session_state.pdf_path,
                        question
                    )
                    answer = response["answer"]
                    sources = response["source_chunks"]

                    st.markdown(answer)

                    with st.expander("📄 View Source Chunks"):
                        for i, chunk in enumerate(sources):
                            st.caption(f"*Source {i+1}:* {chunk}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")