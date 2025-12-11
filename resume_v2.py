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

short_summary_template = """
You are an expert in concise communication.
Based on the following text: "{text}"

TASK:
Write a summary of this book in **a single dense and punchy paragraph** (approx. 150-200 words).
Do not use bullet points.
Capture the absolute essence: the central problem, the solution proposed by the author, and the transformation promised to the reader.
The tone must be engaging.
"""

blinkist_template = """
You are an expert in literary synthesis.
Book text: "{text}"

TASK:
Create a detailed **chapter-by-chapter** summary.
Structure the response for each major section:
1. **Concept Title**
2. **The Key Idea**: What is the main lesson?
3. **Analysis**: Key arguments and examples.
4. **Action**: What the reader must do.

Format: Clean Markdown.
"""

long_summary_template = """
You are a high-level consultant.
Book text: "{text}"

TASK:
Write a comprehensive "Executive Summary" of about **1000 words** (2 pages).
Required structure:
1. **Central Thesis**
2. **Context & Audience**
3. **The 3 Main Pillars** (In-depth development)
4. **Key Case Studies**
5. **Conclusion & Impact**

Tone: Professional and analytical.
"""

chain_short = ChatPromptTemplate.from_template(short_summary_template) | model | StrOutputParser()
chain_blinkist = ChatPromptTemplate.from_template(blinkist_template) | model | StrOutputParser()
chain_long = ChatPromptTemplate.from_template(long_summary_template) | model | StrOutputParser()

for pdf_loader in loaders:
    try:
        file_name = os.path.basename(pdf_loader.file_path)
        base_name = os.path.splitext(file_name)[0]


        documents = pdf_loader.load()
        full_text = "\n".join([doc.page_content for doc in documents])
        
        short_result = chain_short.invoke({"text": full_text})
        with open(f"{base_name}_1_flash.md", "w", encoding="utf-8") as f:
            f.write(short_result)

        blinkist_result = chain_blinkist.invoke({"text": full_text})
        with open(f"{base_name}_2_chapters.md", "w", encoding="utf-8") as f:
            f.write(blinkist_result)

        long_result = chain_long.invoke({"text": full_text})
        with open(f"{base_name}_3_summary.md", "w", encoding="utf-8") as f:
            f.write(long_result)



    except Exception as e:
        print(f"Error on {file_name}: {e}")
