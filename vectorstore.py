from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embedding)

def ingest_documents():
    loader = TextLoader("data/tickets.txt")
    docs = loader.load()
    splits = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    db.add_documents(splits)

def get_vectorstore():
    return db