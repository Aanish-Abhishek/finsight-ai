import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents

def get_document_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    return chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = PromptTemplate.from_template("""
    You are a financial document analyst.
    Answer the question using ONLY the context provided below.
    If the answer is not in the context, say "I couldn't find this in the document."
    Always mention which part of the document supports your answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def execute_chain(pdf_path, question):
    documents = get_pdf_content(pdf_path)
    chunks = get_document_chunks(documents)
    vector_store = get_embeddings(chunks)
    chain, retriever = get_chain(vector_store)

    # Get answer
    answer = chain.invoke(question)

    # Get source chunks separately
    source_docs = retriever.invoke(question)
    source_chunks = [
        f"Page {doc.metadata['page'] + 1}: {doc.page_content[:200]}..."
        for doc in source_docs
    ]

    return {
        "answer": answer,
        "source_chunks": source_chunks
    }