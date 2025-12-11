import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3
)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

pdf_paths = [
    "GenAIbooks/books/atomic_habits.pdf",
    "GenAIbooks/books/grit_the_power.pdf",
    "GenAIbooks/books/indistractable.pdf"
]

all_pages = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    all_pages.extend(pages)

db = FAISS.from_documents(all_pages, embeddings)

rag_template = """
You are an expert library assistant specializing in self-improvement books.
You have access to "Atomic Habits", "Grit", and "Indistractable".

Use the following pieces of context to answer the user's question.
If the context doesn't contain the answer, just say you don't know. 
Do not make up an answer.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=rag_template, 
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt}
)

while True:
    q = input("Enter your question (or 'exit' to quit): ")
    if q.lower() == "exit":
        break
    
    try:
        response = qa_chain.invoke(q)
        print(response['result'])
    except Exception as e:
        print(e)