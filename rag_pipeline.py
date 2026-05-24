import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def get_pdf_content(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    return documents

def get_document_chunks(documents):
    splitter = RecurssiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    return chunks

def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store

def get_qa_chain(vector_store):
    llm = ChatOpenAI(ChatOpenAI(model="gpt-4o-mini", temperature=0.5))
    
    prompt_template = """Your are a finanical document analyst.
                    Answer the question using only the context provided below.
                    If the answer is not contained within the context, say you don't know.
                    Always mention which part of the document supports your answer.
                    
                    Context: {context}
                    Question: {question}
                    Answer:
                    """

    prompt = PromptTemplate.from_template(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain

def execute_chain(pdf_path, question):
    doc_content = get_pdf_content(pdf_path)
    chunks = get_document_chunks(doc_content)
    embeddings = get_embeddings(chunks)
    qa_chain = get_qa_chain(embeddings)
    result = qa_chain.invoke({"question" : question})

    return{
        "answer" : result["result"],
        "source_chunks": [
            f"Page {doc.metadata['page']+1: {doc.page_content[:200]}}..."
            for doc in result["source_documents"]
        ]
    }