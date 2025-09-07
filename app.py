# filename: app.py
import os
import PyPDF2
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from groq import Groq

# ✅ Load Groq client (set your API key as env variable)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

# 📖 Extract text from PDF
def extract_pdf_text(file) -> str:
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


@app.post("/generate-questions")
async def generate_questions(query: str = Form(...), pdf: UploadFile = File(...)):
    try:
        # Read uploaded PDF
        pdf_text = extract_pdf_text(pdf.file)

        # 🔥 Call Groq model
        chat_completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # free Groq model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates educational questions based on PDF content."},
                {"role": "user", "content": f"""
Here is the PDF content (may be truncated):
{pdf_text[:15000]}

Now, answer this query: {query}
"""}
            ],
            temperature=0.3,
            max_tokens=1200
        )

        return JSONResponse({
            "success": True,
            "questions": chat_completion.choices[0].message.content
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        })
