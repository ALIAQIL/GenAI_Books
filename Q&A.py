import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=None
)

loaders = [
    PyPDFLoader("GenAIbooks/books/atomic_habits.pdf"),
    PyPDFLoader("GenAIbooks/books/grit_the_power.pdf"),
    PyPDFLoader("GenAIbooks/books/indistractable.pdf")
]

QandA_template = """
You are an expert in concise communication.
Based on the following text: "{text}"

TASK:
read and understand very good the book
Write a question and answer of this book in 20 questions 
Do not use bullet points.
Capture the absolute essence: the most important parts in the book
"""


chain_QandA = ChatPromptTemplate.from_template(QandA_template) | model | StrOutputParser()

for pdf_loader in loaders:
    try:
        file_name = os.path.basename(pdf_loader.file_path)
        base_name = os.path.splitext(file_name)[0]


        documents = pdf_loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        
        short_result = chain_QandA.invoke({"text": full_text})
        with open(f"{base_name}_QandA.md", "w", encoding="utf-8") as f:
            f.write(short_result)
    except Exception as e:
        print(f"Error on {file_name}: {e}")
