import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    max_output_tokens=None
)

loaders = [
    TextLoader("GenAIbooks/resume/atomic_habits_summary.txt"),
    TextLoader("GenAIbooks/resume/grit_the_power_summary.txt"),
    TextLoader("GenAIbooks/resume/indistractable_summary.txt")
]

trad_french_template = """
You are an expert in concise communication.
Based on the following text: "{text}"

TASK:
read and understand very good the resume
Write an translation in french
Do not use bullet points.
Capture the absolute essence: translate like you are the goat translater
"""

trad_arabic_template = """
You are an expert in concise communication.
Based on the following text: "{text}"

TASK:
read and understand very good the resume
Write an translation in arabic
Do not use bullet points.
Capture the absolute essence: translate like you are the goat translater
"""


chain_trad_fr = ChatPromptTemplate.from_template(trad_french_template) | model | StrOutputParser()
chain_trad_ar = ChatPromptTemplate.from_template(trad_arabic_template) | model | StrOutputParser()

for txt_loader in loaders:
    try:
        file_name = os.path.basename(txt_loader.file_path)
        base_name = os.path.splitext(file_name)[0]


        documents = txt_loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        
        resume_fr = chain_trad_fr.invoke({"text": full_text})
        with open(f"{base_name}_trad_fr.md", "w", encoding="utf-8") as f:
            f.write(resume_fr)
        
        resume_ar = chain_trad_ar.invoke({"text": full_text})
        with open(f"{base_name}_trad_ar.md", "w", encoding="utf-8") as f:
            f.write(resume_ar)
    except Exception as e:
        print(f"Error on {file_name}: {e}")
