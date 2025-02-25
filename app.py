import os
import shutil
import fitz  # PyMuPDF for PDF text extraction
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form

# 🔹 Configure Google AI Studio API
GOOGLE_AI_STUDIO_API_KEY = "AIzaSyDeNdQRgQvj7ZPP7Qc3BncwM1VdzMP5eGw"
genai.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)

app = FastAPI()

# 🔹 Function to Extract Text from PDF Resume
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF resume file."""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc]).strip()
        return text
    except Exception as e:
        return str(e)

# 🔹 Resume Screening API (Supports Local Files & Uploads)
@app.post("/screen_resume/")
async def screen_resume(
    file: UploadFile = None, 
    job_description: str = Form(...), 
    local_file_path: str = Form(None)
):
    """
    Extracts text from an uploaded PDF or a local file and analyzes the resume fit.
    - If `file` is provided, it will process the uploaded file.
    - If `local_file_path` is provided, it will process the local file instead.
    """

    try:
        if file:
            # Save uploaded file to temporary path
            temp_pdf_path = f"temp_{file.filename}"
            with open(temp_pdf_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        elif local_file_path:
            if not os.path.exists(local_file_path):
                return {"error": "Local file not found."}
            temp_pdf_path = local_file_path
        else:
            return {"error": "No file provided."}

        # Extract resume text
        resume_text = extract_text_from_pdf(temp_pdf_path)

        # AI Model Prompt
        prompt = f"""
        You are an AI that screens resumes for a 'Software Engineer' role.

        Resume:
        {resume_text}

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

        # Cleanup temporary file (only if uploaded)
        if file:
            os.remove(temp_pdf_path)

        return {"resume_screening_result": response.text}

    except Exception as e:
        return {"error": str(e)}

# 🔹 Sentiment Analysis API Endpoint
@app.post("/analyze_sentiment/")
async def analyze_employee_feedback(feedback: str = Form(...)):
    """Uses Google's Gemini API to analyze employee sentiment."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Analyze the sentiment of this employee feedback: {feedback}")

    return {"sentiment_analysis_result": response.text}

# 🔹 Root Endpoint
@app.get("/")
def home():
    return {"message": "Welcome to AI-powered HR API"}





import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT environment variable
    uvicorn.run(app, host="0.0.0.0", port=port)
