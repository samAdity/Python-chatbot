import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = "sk-Wr5VzIVOwRoIyzTkQTjiaLQ6lSc84" #Pass your key here


#Upload PDF files
st.header("My first Chatbot")


with  st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader(" Upload a PDf file and start asking questions", type="pdf")


#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)


#Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)




    # generating embedding
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


    # creating vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)


    # get user question
    user_question = st.text_input("Type Your question here")


    # do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)


        #define the LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 1000,
            model_name = "gpt-3.5-turbo"
        )


        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)







2 nd code 

# main.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Ollama
from langdetect import detect
from googletrans import Translator

import tempfile
import os

# ========== SETUP ==========
st.set_page_config(page_title="Galaxy Chatbot 🤖🌌", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ========== CUSTOM UI STYLING ==========
st.markdown("""
    <style>
        body {
            background: radial-gradient(circle at top, #1b2735 0%, #090a0f 100%);
            color: white;
        }
        .stApp {
            background-color: #000;
        }
        .robot-avatar {
            width: 90px;
        }
        .chat-bubble {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 GalaxyBot – Your Multilingual PDF Chat Assistant")
st.markdown("Chat with multiple PDFs, in any language. Built with 🚀 Ollama + 🧠 Langchain")

# ========== MULTI-PDF UPLOAD ==========
uploaded_files = st.sidebar.file_uploader("📂 Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)
process_btn = st.sidebar.button("📄 Process Documents")

# ========== TEXT EXTRACTION ==========
def extract_text_from_pdfs(files):
    all_text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                all_text += text
    return all_text

# ========== TEXT CHUNKING + VECTOR STORE ==========
def build_vector_store_from_text(raw_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Local + Fast
    return FAISS.from_texts(texts, embeddings)

# ========== MULTILINGUAL TRANSLATION ==========
translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text):
    lang = detect_language(text)
    if lang != "en":
        return translator.translate(text, src=lang, dest="en").text
    return text

def translate_to_original(text, original_lang):
    if original_lang != "en":
        return translator.translate(text, src="en", dest=original_lang).text
    return text

# ========== OLLAMA LLM CALL ==========
llm = Ollama(model="llama2")

# ========== PROCESS BUTTON ==========
if process_btn and uploaded_files:
    with st.spinner("🔍 Reading and embedding documents..."):
        raw_text = extract_text_from_pdfs(uploaded_files)
        vector_store = build_vector_store_from_text(raw_text)
        st.session_state.vector_store = vector_store
    st.success("✅ Documents processed and embedded!")

# ========== CHAT INTERFACE ==========
user_input = st.chat_input("💬 Ask me something...")
if user_input and st.session_state.vector_store:
    original_lang = detect_language(user_input)
    english_query = translate_to_english(user_input)

    relevant_docs = st.session_state.vector_store.similarity_search(english_query)
    chain = load_qa_chain(llm, chain_type="stuff")
    response = chain.run(input_documents=relevant_docs, question=english_query)

    translated_response = translate_to_original(response, original_lang)

    # Add to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("GalaxyBot 🤖", translated_response))

# ========== DISPLAY CHAT HISTORY ==========
for sender, msg in st.session_state.chat_history:
    with st.container():
        st.markdown(f"**{sender}:**")
        st.markdown(f"<div class='chat-bubble'>{msg}</div>", unsafe_allow_html=True)

