import google.generativeai as genai
import fitz  # PyMuPDF for PDF text extraction
from fastapi import FastAPI, File, UploadFile, Form
import shutil
import os

# 🔹 Configure Google AI Studio API
GOOGLE_AI_STUDIO_API_KEY = "AIzaSyDeNdQRgQvj7ZPP7Qc3BncwM1VdzMP5eGw"
genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)

app = FastAPI()

# 🔹 Function to Extract Text from PDF Resume
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF resume file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# 🔹 Resume Screening API Endpoint
@app.post("/screen_resume")
async def screen_resume(file: UploadFile = File(...), job_description: str = Form(...)):
    """Extracts text from an uploaded PDF and analyzes resume fit."""
    
    # Save the uploaded file
    temp_pdf_path = f"temp_{file.filename}"
    with open(temp_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract resume text
    resume_text = extract_text_from_pdf(temp_pdf_path)

    # AI Model Prompt
    prompt = f"""
    You are an AI that screens resumes for a "Software Engineer" role.

    Resume: {resume_text}

    Compare it with this job description:
    {job_description}

    Provide structured output:
    - Candidate Name
    - Key Skills
    - Experience Level (Years)
    - Education
    - Match Score (0-100)
    - Missing Skills (if any)
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    # Cleanup
    os.remove(temp_pdf_path)

    return {"resume_screening_result": response.text}

# 🔹 Sentiment Analysis API Endpoint
@app.post("/analyze_sentiment")
async def analyze_employee_feedback(feedback: str = Form(...)):
    """Uses Google's Gemini API to analyze employee sentiment."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Analyze the sentiment of this employee feedback: {feedback}")

    return {"sentiment_analysis_result": response.text}

# 🔹 Root Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to AI-powered HR API"}
