import os
import google.generativeai as gen_ai
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import pandas as pd
import docx
from time import sleep
from logger import setup_logger

# Set up logger
logger = setup_logger('util_logger', 'logs/util.log')

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
logger.info("GOOGLE_API_KEY loaded.")
gen_ai.configure(api_key=GOOGLE_API_KEY)

system_prompt = """
                    You are "Jarvis," a highly intelligent and friendly AI assistant. You excel at understanding complex documents, identifying hidden patterns, and providing valuable, user-friendly solutions to any query. As an expert document analyzer, your mission is to help the user unlock secret insights, offer actionable advice, and present findings in a clear, concise manner.

                    Persona:
                    1. Friendly: Always respond with warmth and professionalism, making the user feel at ease.
                    2. Intelligent: Provide detailed, accurate, and advanced solutions to even the most complex queries.
                    3. Analytical: Dive deep into the content of documents, analyzing text to uncover hidden patterns, trends, and key insights that are not immediately obvious.
                    4. Solution-Oriented: Focus on offering meaningful, actionable answers that directly address the user's goals and problems.

                    ### Introduction Example:
                    Hello! I'm Vs, your friendly and intelligent AI assistant, specialized in document analysis. I’m here to help you unlock hidden patterns, discover valuable insights, and provide clear, actionable solutions from any document you upload. Whether it's complex data, research papers, or business reports, I’ve got you covered!

                    Feel free to ask me anything, or upload a document for a detailed analysis, and let’s uncover some insights together!

                    ### User Task Example:
                    The user uploads a document and asks Vs to analyze the content for hidden patterns and insights. They are seeking a clear recommendation based on the analysis.

                    ### Response Example:
                    - Begin by acknowledging the document and summarizing its main themes.
                    - Move into deeper analysis, highlighting any patterns, anomalies, or insights you’ve uncovered.
                    - Conclude by offering practical solutions or recommendations based on the document’s content.
                """

model = gen_ai.GenerativeModel(model_name="gemini-1.5-flash", system_instruction=system_prompt)

# Load Sentence-BERT model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
logger.info("Sentence-BERT model loaded.")

def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    logger.info(f"Extracted text from PDF, number of pages: {len(doc)}")
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    logger.info("Extracted text from DOCX.")
    return text

def extract_text_from_csv(file):
    df = pd.read_csv(file)
    logger.info(f"Extracted text from CSV, number of rows: {len(df)}")
    return df.to_string()

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    logger.info(f"Extracted text from Excel, number of rows: {len(df)}")
    return df.to_string()

def get_conversation_history(chat_history):
    history = ""
    for message_type, text in chat_history:
        if message_type == "user":
            history += f"User: {text}\n"
        else:
            history += f"Assistant: {text}\n"
    logger.info("Retrieved conversation history.")
    return history

def generate_ai_response(input_with_memory):
    logger.info("Generating AI response.")
    response = model.generate_content(input_with_memory)
    if response.candidates and len(response.candidates) > 0:
        generated_content = response.candidates[0].content.parts[0].text
        full_response = ""
        for chunk in generated_content.split('.'):
            if chunk.strip():
                full_response += chunk.strip() + '.\n\n'  # Add double newlines for better separation
                sleep(0.5)  # Pause to simulate streaming effect

        # Strip out any "Assistant:" labels from the generated content
        full_response = full_response.replace('Assistant:', '').strip()
        logger.info("AI response generated successfully.")
        return full_response
    logger.error("Failed to generate AI response.")
    return None
