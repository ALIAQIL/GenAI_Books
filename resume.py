import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash"
                               , temperature=1.3
                               , max_output_tokens=None
                               )

loaders = [PyPDFLoader("GenAIbooks/books/atomic_habits.pdf")
           ,PyPDFLoader("GenAIbooks/books/grit_the_power.pdf")
           ,PyPDFLoader("GenAIbooks/books/indistractable.pdf")]

summary_template = """
    You are a world-class literary analyst and productivity expert. 
    Your goal is to distill the text below into a high-impact summary that maximizes value for the reader.

    Text to analyze:
    "{text}"

    Please structure your response as follows:

    1. **The One-Sentence Hook**: What is the single most important argument this text makes?
    2. **Core Concepts (The "What")**: Identify the 3 most critical mental models, frameworks, or definitions introduced.
    3. **Actionable Takeaways (The "How")**: Extract specific, practical steps the reader can apply immediately. Use bullet points.
    4. **Target Audience**: Who specifically needs to hear this message right now?

    Tone: Insightful, direct, and professional. Avoid fluff.
"""

prompt = ChatPromptTemplate.from_template(summary_template)

chain = prompt | model | StrOutputParser()

for pdf_loader in loaders:
    file_name = os.path.basename(pdf_loader.file_path)
    print(f"Processing {file_name}...")
    
    documents = pdf_loader.load()
    
    full_text = "\n".join([doc.page_content for doc in documents])
    
    try:
        result = chain.invoke({"text": full_text})
        
        output_filename = f"{os.path.splitext(file_name)[0]}_summary.txt"
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(result)
            
        print(f"Finished {output_filename}")
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")



