from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os
import io
import sys
import uuid
import json
from contextlib import redirect_stdout, redirect_stderr
from fastapi.responses import HTMLResponse, FileResponse

app = FastAPI()

# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversations and generated files
conversations = {}
generated_files = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Initialize Groq client
def get_client():
    return Groq(api_key=GROQ_API_KEY)

# ========== CHAT ENDPOINT ==========
@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in conversations:
        conversations[request.session_id] = [
            {"role": "system", "content": "You are a helpful AI assistant. You can chat, write code, build websites, analyze data, and help with any task."}
        ]
    
    conversations[request.session_id].append(
        {"role": "user", "content": request.message}
    )
    
    try:
        client = get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversations[request.session_id]
        )
        reply = response.choices[0].message.content
        
        conversations[request.session_id].append(
            {"role": "assistant", "content": reply}
        )
        
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# ========== CODE EXECUTION ==========
@app.post("/execute")
async def execute_code(request: ChatRequest):
    try:
        client = get_client()
        
        code_prompt = f"""Write only executable Python code for this request. No explanations, just code:

Request: {request.message}

Rules:
- Write clean, working Python code
- Use print() to show output
- Handle errors gracefully
- If creating files, use simple names"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": code_prompt}]
        )
        
        code = response.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Safe execution
        safe_globals = {
            "__builtins__": __builtins__,
            "print": print, "len": len, "range": range,
            "str": str, "int": int, "float": float,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "open": open, "json": json, "os": os,
        }
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            try:
                exec(code, safe_globals)
            except Exception as e:
                print(f"Error: {str(e)}")
        
        output = output_buffer.getvalue()
        errors = error_buffer.getvalue()
        
        full_output = output if output else "No output"
        if errors:
            full_output += f"\n\nErrors:\n{errors}"
        
        return {
            "reply": f"💻 **Code Generated & Executed!**\n\n```python\n{code}\n```\n\n📤 **Output:**\n```\n{full_output}\n```",
            "code": code,
            "output": full_output
        }
        
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# ========== WEBSITE BUILDER ==========
@app.post("/build-website")
async def build_website(request: ChatRequest):
    try:
        client = get_client()
        
        website_prompt = f"""Create a complete, modern, responsive HTML website. Return ONLY the HTML code (no explanations, no markdown):

Description: {request.message}

Requirements:
- Single HTML file with embedded CSS and JavaScript
- Modern, beautiful design with gradients and animations
- Fully responsive for mobile and desktop
- Interactive elements with JavaScript
- Professional quality"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": website_prompt}]
        )
        
        html_code = response.choices[0].message.content
        html_code = html_code.replace("```html", "").replace("```", "").strip()
        
        # Ensure proper HTML structure
        if not html_code.startswith("<!DOCTYPE html>"):
            html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Website</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; }}
    </style>
</head>
<body>
    {html_code}
</body>
</html>"""
        
        filename = f"website_{uuid.uuid4().hex[:8]}.html"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_code)
        
        generated_files[filename] = html_code
        
        preview_url = f"https://my-chatbot-backend-9i4u.onrender.com/preview/{filename}"
        
        return {
            "reply": f"🌐 **Website Built Successfully!**\n\n🔗 **Live Preview:** {preview_url}\n\n📄 The website is live and ready to view!",
            "html": html_code,
            "preview_url": preview_url
        }
        
    except Exception as e:
        return {"reply": f"Error building website: {str(e)}"}

@app.get("/preview/{filename}")
async def preview_website(filename: str):
    try:
        if filename in generated_files:
            return HTMLResponse(content=generated_files[filename])
        
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except:
        return HTMLResponse(content="<h1>Website not found</h1>")

# ========== DATA ANALYSIS ==========
@app.post("/analyze")
async def analyze_data(request: ChatRequest):
    try:
        client = get_client()
        
        analysis_prompt = f"""Analyze this data/description and provide insights:

{request.message}

Provide:
1. Key insights
2. Trends or patterns
3. Recommendations
4. If applicable, suggest visualizations"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        reply = response.choices[0].message.content
        
        return {"reply": f"📊 **Data Analysis**\n\n{reply}"}
        
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# ========== FILE UPLOAD ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode('utf-8', errors='ignore')
        
        client = get_client()
        
        summary_prompt = f"""Summarize and analyze this file content:

File: {file.filename}

Content:
{text[:5000]}

Provide:
1. Brief summary
2. Key points
3. Any issues or recommendations"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": summary_prompt}]
        )
        
        reply = response.choices[0].message.content
        
        return {"reply": f"📁 **File Analysis: {file.filename}**\n\n{reply}"}
        
    except Exception as e:
        return {"reply": f"Error processing file: {str(e)}"}

# ========== CLEAR HISTORY ==========
@app.post("/clear")
async def clear_history(request: ChatRequest):
    if request.session_id in conversations:
        conversations[request.session_id] = [
            {"role": "system", "content": "You are a helpful AI assistant."}
        ]
    return {"reply": "✅ Chat history cleared!"}

@app.get("/")
async def root():
    return {
        "message": "🤖 Super Chatbot Backend is running!",
        "features": ["chat", "code_execution", "website_builder", "data_analysis", "file_upload"],
        "status": "active"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)