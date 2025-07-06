from app import load_webpage, get_chat_response
import streamlit as st

st.set_page_config(page_title="RAG Langchain based Chatbot", page_icon=":robot_face:")

st.title("RAG, Langchain Based Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

URL = st.text_input("Enter the website URL you want to query")

if URL and ("db" not in st.session_state or st.session_state.get("last_url") != URL):
    with st.spinner("Loading documents from given URL..."):
        st.session_state.db = load_webpage(URL)
        st.session_state.last_url = URL
        st.success("Loaded the documents from the URL and created the vector store.")

query = st.chat_input("Ask me anything...")
if query and "db" in st.session_state:
        get_chat_response(query, st.session_state.db)
        