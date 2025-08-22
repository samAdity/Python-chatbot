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







# 2 nd code 

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




3 rd file

# main.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.prompts import SystemMessagePromptTemplate
from langdetect import detect
from googletrans import Translator

import os

# ========== CONFIGURATION ==========
OPENAI_API_KEY = "your-openai-api-key-here"  # <-- Replace this with your actual key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

st.set_page_config(page_title="Galaxy Chatbot 🤖🌌", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "top_k" not in st.session_state:
    st.session_state.top_k = 3
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 150

# ========== UI STYLING ==========
st.markdown("""
    <style>
        .stApp { background-color: #0b0c2a; color: #ffffff; }
        .chat-bubble {
            background-color: #222244;
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0px;
        }
    </style>
""", unsafe_allow_html=True)

st.title(":milky_way: GalaxyBot - Multilingual PDF Chat Assistant")

# ========== PDF UPLOAD ==========
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded_files = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.chunk_size)
    st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 500, st.session_state.chunk_overlap)
    st.session_state.top_k = st.slider("Top K Documents to Retrieve", 1, 10, st.session_state.top_k)
    process_btn = st.button("🔍 Process Documents")

# ========== TRANSLATION UTILS ==========
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

# ========== TEXT EXTRACTION ==========
def extract_text_from_pdfs(files):
    text = ""
    for file in files:
        pdf = PdfReader(file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ========== EMBEDDING & VECTOR DB ==========
def build_vector_store(text, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(chunks, embeddings)

# ========== PROCESS PDF ==========
if process_btn and uploaded_files:
    with st.spinner("Embedding documents, please wait..."):
        raw_text = extract_text_from_pdfs(uploaded_files)
        vs = build_vector_store(raw_text, st.session_state.chunk_size, st.session_state.chunk_overlap)
        st.session_state.vector_store = vs
    st.success("Documents embedded successfully!")

# ========== QA Chain (ReAct Agent Style + Prompt Optimization) ==========
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3, max_tokens=1000)

system_prompt = SystemMessagePromptTemplate.from_template("""
You are GalaxyBot, a multilingual, helpful assistant. Use only information from the provided documents to answer accurately.
If unsure, say "I'm not sure based on the provided documents."
""")

# ========== CHAT INTERFACE ==========
user_input = st.chat_input("Ask your question in any language...")
if user_input and st.session_state.vector_store:
    original_lang = detect_language(user_input)
    english_query = translate_to_english(user_input)

    docs = st.session_state.vector_store.similarity_search(english_query, k=st.session_state.top_k)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k": st.session_state.top_k}),
        return_source_documents=True
    )

    with st.spinner("Generating response..."):
        result = qa({"query": english_query})
        answer = result["result"]
        translated = translate_to_original(answer, original_lang)

        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("GalaxyBot 🤖", translated))

# ========== DISPLAY CHAT ==========
for sender, msg in st.session_state.chat_history:
    with st.container():
        st.markdown(f"**{sender}:**")
        st.markdown(f"<div class='chat-bubble'>{msg}</div>", unsafe_allow_html=True)



4 th 

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------- SETUP ----------
st.set_page_config(page_title="GalaxyBot 🤖", layout="centered")

# ---------- GALAXY BACKGROUND ----------
st.markdown("""
<style>
body {
    background: url("https://img.freepik.com/free-vector/galaxy-background-with-gradient-colors_23-2149241204.jpg");
    background-size: cover;
    background-attachment: fixed;
    color: white;
}
.sidebar .sidebar-content {
    background: #0b1a2a;
    color: white;
}
.chat-box {
    background-color: rgba(0,0,0,0.6);
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------- 3D ROBOT IMAGE ----------
st.image("https://i.ibb.co/2ytnsqt/robot-3d.gif", width=120, caption="GalaxyBot")

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📄 Upload PDF File")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    st.markdown("---")
    st.markdown("🌌 Powered by OpenAI + Langchain")

# ---------- EXTRACT PDF TEXT ----------
pdf_text = ""
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    for page in pdf_reader.pages[:3]:  # Only first 3 pages for speed
        content = page.extract_text()
        if content:
            pdf_text += content

# ---------- QUESTION INPUT ----------
st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
question = st.text_input("💬 Ask me a question:")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- RESPONSE ----------
if question:
    with st.spinner("🧠 Thinking..."):
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # <-- Replace with your API key
        llm = OpenAI(temperature=0.6, model_name="gpt-3.5-turbo")

        if pdf_text.strip():
            prompt_text = f"""
Answer the question based on the following PDF content (only first few pages used):

PDF:
{pdf_text[:3000]}

Question: {question}
"""
        else:
            prompt_text = f"Answer this general question: {question}"

        prompt = PromptTemplate(input_variables=["input"], template="{input}")
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(prompt_text)

        st.success(response)
