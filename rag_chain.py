from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from vectorstore import get_vectorstore

llm = Ollama(model="mistral")
db = get_vectorstore()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())