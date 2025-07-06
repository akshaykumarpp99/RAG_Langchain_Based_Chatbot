import os
import bs4
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import streamlit as st

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.2
    )

def load_webpage(URL):
    if URL:

        db_path = f"faiss_{hash(URL)}.index"
        if os.path.exists(db_path):
            db = FAISS.load_local(
                db_path, 
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
                allow_dangerous_deserialization=True
                )
            
            return db
        loader = WebBaseLoader(URL)
        documents=loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        db = FAISS.from_documents(docs, embeddings)
        db.save_local(db_path)
        return db   

def get_chat_response(query, db):     
    if query:                      
        for msf in st.session_state.chat_history:
                with st.chat_message(msf["role"]):
                    st.markdown(msf["message"]) 
                                                                      
        with st.spinner("Thinking..."):
            vectorsearch_result = db.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in vectorsearch_result])
            prompt = f"""
            You are an assistant. Use the following context to answer the question.

            Context: {context}

            Question: {query}   
            """
            response = llm.invoke(prompt)
            
            with st.chat_message("user"):
                st.markdown(query)
            
            with st.chat_message("assistant"):
                st.markdown("LLM Response: " + response.content)
            st.session_state.chat_history.append({"role": "user", "message": query})
            st.session_state.chat_history.append({"role": "assistant", "message": response.content})
    
