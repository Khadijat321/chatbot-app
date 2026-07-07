from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from groq import Groq
import os
import io
import sys
import uuid
import json
import base64
import requests
import sqlite3
import hashlib
import time
import re
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, List, Dict, Any
import traceback

# Data science & visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO

# Document processing
try:
    import PyPDF2
    PYPDF_AVAILABLE = True
except:
    PYPDF_AVAILABLE = False

try:
    from openpyxl import load_workbook
    OPENPYXL_AVAILABLE = True
except:
    OPENPYXL_AVAILABLE = False

# Text-to-speech
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except:
    GTTS_AVAILABLE = False

# FAISS for RAG
try:
    import faiss
    FAISS_AVAILABLE = True
except:
    FAISS_AVAILABLE = False

app = FastAPI(title="Omni AI Backend", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIGURATION ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Store data
conversations: Dict[str, List[Dict]] = {}
generated_files: Dict[str, str] = {}
user_memories: Dict[str, Dict] = {}

# RAG document store
rag_documents: Dict[str, List[Dict]] = {}

# Initialize SQLite for persistent memory
DB_PATH = "omni_ai_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp REAL,
            mode TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            filename TEXT,
            content TEXT,
            timestamp REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS generated_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            type TEXT,
            content TEXT,
            url TEXT,
            timestamp REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ========== MODELS ==========
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: str = "chat"
    image_data: Optional[str] = None

class AgentRequest(BaseModel):
    message: str
    session_id: str = "default"
    tools: List[str] = Field(default_factory=lambda: ["search", "code", "weather", "news"])

class TranslateRequest(BaseModel):
    message: str
    target_language: str = "en"
    session_id: str = "default"

class SentimentRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChartRequest(BaseModel):
    message: str
    session_id: str = "default"

class WeatherRequest(BaseModel):
    message: str
    session_id: str = "default"

class NewsRequest(BaseModel):
    message: str
    session_id: str = "default"

class SearchRequest(BaseModel):
    message: str
    session_id: str = "default"

class TTSRequest(BaseModel):
    message: str
    lang: str = "en"
    session_id: str = "default"

class ImageRequest(BaseModel):
    message: str
    session_id: str = "default"

class WebsiteRequest(BaseModel):
    message: str
    session_id: str = "default"

class AnalyzeRequest(BaseModel):
    message: str
    session_id: str = "default"

class VideoRequest(BaseModel):
    message: str
    session_id: str = "default"

class RAGQueryRequest(BaseModel):
    message: str
    session_id: str = "default"

class ClearRequest(BaseModel):
    session_id: str = "default"

# ========== HELPER FUNCTIONS ==========
def get_client():
    return Groq(api_key=GROQ_API_KEY)

def save_message(session_id: str, role: str, content: str, mode: str = "chat"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations (session_id, role, content, timestamp, mode) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, time.time(), mode)
    )
    conn.commit()
    conn.close()

def get_conversation_history(session_id: str, limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using Tavily (free tier) with DuckDuckGo fallback"""
    # Try Tavily first (AI-optimized search)
    if TAVILY_API_KEY:
        try:
            url = "https://api.tavily.com/search"
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": num_results,
                "include_answer": True
            }

            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            data = resp.json()

            answer = data.get("answer", "")
            output = f"\n**Web Search Results for '{query}':**\n"

            if answer:
                output += f"\n**Quick Answer:** {answer}\n"

            for i, result in enumerate(data.get("results", [])[:num_results], 1):
                title = result.get("title", "No title")
                content = result.get("content", "")
                url_link = result.get("url", "")

                output += f"\n{i}. **{title}**\n"
                if content:
                    output += f"   {content[:300]}{'...' if len(content) > 300 else ''}\n"
                if url_link:
                    output += f"   {url_link}\n"

            return output

        except Exception as e:
            pass  # Fallback to DuckDuckGo

    # DuckDuckGo fallback (completely free, no key needed)
    try:
        url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=10)

        titles = re.findall(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', resp.text)
        snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', resp.text)

        output = f"\n**Web Search Results for '{query}':**\n"
        for i in range(min(num_results, len(titles))):
            title = re.sub(r'<[^>]+>', '', titles[i])
            snippet = re.sub(r'<[^>]+>', '', snippets[i]) if i < len(snippets) else ""
            output += f"\n{i+1}. **{title}**\n   {snippet}\n"

        return output if titles else f"\nSearched for '{query}'. No results found.\n"

    except Exception as e:
        return f"\nWeb search unavailable. Add TAVILY_API_KEY for better search.\n"

def get_weather(city: str) -> str:
    """Get weather data from OpenWeatherMap"""
    if not OPENWEATHER_API_KEY:
        return "Weather API not configured. Add OPENWEATHER_API_KEY."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("cod") != 200:
            return f"Weather error: {data.get('message', 'Unknown error')}"

        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        desc = data["weather"][0]["description"]
        wind = data["wind"]["speed"]

        return f"""Weather in {city.title()}
Temperature: {temp}C (feels like {feels_like}C)
Humidity: {humidity}%
Conditions: {desc.title()}
Wind: {wind} m/s"""
    except Exception as e:
        return f"Weather fetch failed: {str(e)}"

def get_news(query: str = "technology", num: int = 5) -> str:
    """Get news from NewsData.io"""
    if not NEWSDATA_API_KEY:
        return "News API not configured. Add NEWSDATA_API_KEY."

    try:
        url = f"https://newsdata.io/api/1/latest?apikey={NEWSDATA_API_KEY}&q={requests.utils.quote(query)}&language=en&size={num}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        if data.get("status") != "success":
            return f"News error: {data.get('results', {}).get('message', 'Unknown error')}"

        articles = data.get("results", [])
        output = f'\n**Latest News on "{query}":**\n'

        for i, art in enumerate(articles[:num], 1):
            title = art.get("title", "No title")
            desc = art.get("description", "") or art.get("content", "")[:200]
            link = art.get("link", "")
            source = art.get("source_id", "Unknown")
            pub_date = art.get("pubDate", "")

            output += f"\n{i}. **{title}**\n"
            if desc:
                output += f"   {desc}\n"
            if link:
                output += f"   {link}\n"
            output += f"   Source: {source}"
            if pub_date:
                output += f" | {pub_date}"
            output += "\n"

        return output

    except Exception as e:
        return f"News fetch failed: {str(e)}"

def generate_image(prompt: str, width: int = 1024, height: int = 768) -> str:
    """Generate image using Pollinations.ai (free)"""
    clean = requests.utils.quote(prompt[:400])
    return f"https://image.pollinations.ai/prompt/{clean}?width={width}&height={height}&nologo=true&seed={int(time.time())}&enhance=true"

def generate_chart(chart_type: str, data: dict, title: str = "Chart") -> str:
    """Generate matplotlib chart and return base64"""
    try:
        plt.figure(figsize=(10, 6))

        if chart_type == "bar":
            plt.bar(data.get("labels", []), data.get("values", []), color="#6366f1")
        elif chart_type == "line":
            plt.plot(data.get("labels", []), data.get("values", []), marker="o", color="#6366f1", linewidth=2)
        elif chart_type == "pie":
            plt.pie(data.get("values", []), labels=data.get("labels", []), autopct="%1.1f%%")
        elif chart_type == "scatter":
            plt.scatter(data.get("x", []), data.get("y", []), color="#6366f1", alpha=0.6)
        elif chart_type == "histogram":
            plt.hist(data.get("values", []), bins=data.get("bins", 10), color="#6366f1", alpha=0.7)

        plt.title(title, fontsize=14, fontweight="bold", color="white")
        plt.xlabel(data.get("xlabel", ""), color="#94a3b8")
        plt.ylabel(data.get("ylabel", ""), color="#94a3b8")
        plt.tick_params(colors="#94a3b8")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a0a12")
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        return f"Error generating chart: {str(e)}"

def text_to_speech(text: str, lang: str = "en") -> str:
    """Convert text to speech, return base64 audio"""
    if not GTTS_AVAILABLE:
        return None
    try:
        tts = gTTS(text=text[:500], lang=lang, slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode()
        return f"data:audio/mp3;base64,{audio_b64}"
    except Exception as e:
        return None

def extract_text_from_file(content: bytes, filename: str) -> str:
    """Extract text from various file types"""
    ext = filename.lower().split(".")[-1] if "." in filename else ""

    if ext in ["txt", "md", "csv", "json", "py", "js", "html", "css"]:
        return content.decode("utf-8", errors="ignore")

    elif ext == "pdf" and PYPDF_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(BytesIO(content))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except:
            return "Error reading PDF"

    elif ext in ["xlsx", "xls"] and OPENPYXL_AVAILABLE:
        try:
            wb = load_workbook(BytesIO(content))
            text = ""
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                text += f"\n--- Sheet: {sheet} ---\n"
                for row in ws.iter_rows(values_only=True):
                    text += " | ".join(str(c) if c else "" for c in row) + "\n"
            return text
        except:
            return "Error reading Excel file"

    return content.decode("utf-8", errors="ignore")[:10000]
# ========== SYSTEM PROMPTS ==========
SYSTEM_PROMPTS = {
    "chat": "You are Omni AI, a super-intelligent assistant with access to tools, vision, code execution, web search, image generation, data analysis, and more. Be helpful, concise, and accurate.",
    "code": "You are an expert programmer. Write clean, efficient, well-documented code. Always include error handling.",
    "vision": "You are a computer vision expert. Describe images in detail, identify objects, read text, and answer questions about visual content.",
    "agent": "You are an AI agent that can use tools. When asked a question, determine if you need to search the web, get weather, fetch news, generate code, or create images. Use the appropriate tool and synthesize the results.",
    "analyst": "You are a data scientist. Analyze data thoroughly, identify trends, create insights, and suggest visualizations when helpful."
}

# ========== CORE ENDPOINTS ==========

@app.get("/")
async def root():
    return {
        "name": "Omni AI Backend v2.0",
        "status": "Active",
        "features": [
            "chat", "vision", "code_execution", "website_builder",
            "data_analysis", "file_upload", "image_generation",
            "web_search", "weather", "news", "agent_mode",
            "text_to_speech", "chart_generation", "youtube_creator",
            "translation", "sentiment_analysis", "rag_qa"
        ],
        "models": ["llama-3.1-8b-instant", "llama-3.2-90b-vision-preview", "whisper-large-v3"],
        "endpoints": {
            "chat": "POST /chat",
            "vision": "POST /vision",
            "execute": "POST /execute",
            "build_website": "POST /build-website",
            "analyze": "POST /analyze",
            "upload": "POST /upload",
            "generate_image": "POST /generate-image",
            "search": "POST /search",
            "weather": "POST /weather",
            "news": "POST /news",
            "agent": "POST /agent",
            "tts": "POST /tts",
            "chart": "POST /chart",
            "create_video": "POST /create-video",
            "translate": "POST /translate",
            "sentiment": "POST /sentiment",
            "rag_upload": "POST /rag/upload",
            "rag_query": "POST /rag/query",
            "clear": "POST /clear",
            "history": "GET /history/{session_id}"
        }
    }

# ========== 1. ENHANCED CHAT ==========
@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or "default"

    if session_id not in conversations:
        conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPTS.get(request.mode, SYSTEM_PROMPTS["chat"])}
        ]

    if request.image_data:
        return await vision_chat(request)

    conversations[session_id].append({"role": "user", "content": request.message})
    save_message(session_id, "user", request.message, request.mode)

    try:
        client = get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversations[session_id],
            temperature=0.7,
            max_tokens=4096
        )
        reply = response.choices[0].message.content

        conversations[session_id].append({"role": "assistant", "content": reply})
        save_message(session_id, "assistant", reply, request.mode)

        return {"reply": reply, "mode": request.mode, "session_id": session_id}
    except Exception as e:
        return {"reply": f"Error: {str(e)}", "mode": request.mode}

# ========== 2. VISION AI ==========
@app.post("/vision")
async def vision_chat(request: ChatRequest):
    session_id = request.session_id or "default"

    if not request.image_data:
        return {"reply": "No image provided. Please upload an image."}

    try:
        client = get_client()

        image_content = request.image_data
        if image_content.startswith("data:image"):
            image_content = image_content.split(",")[1]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.message or "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_content}"}}
                ]
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            max_tokens=4096
        )

        reply = response.choices[0].message.content
        save_message(session_id, "user", f"[Image] {request.message}", "vision")
        save_message(session_id, "assistant", reply, "vision")

        return {"reply": reply, "mode": "vision"}
    except Exception as e:
        return {"reply": f"Vision analysis error: {str(e)}"}

# ========== 3. ENHANCED CODE EXECUTION ==========
@app.post("/execute")
async def execute_code(request: ChatRequest):
    try:
        client = get_client()

        code_prompt = f"""Write only executable Python code for this request. No explanations outside code blocks:

Request: {request.message}

Rules:
- Write clean, working Python code
- Use print() to show output
- Handle errors with try/except
- For charts, save to buffer and print base64
- If creating files, use simple names in /tmp/"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": code_prompt}],
            temperature=0.3
        )

        raw_code = response.choices[0].message.content

        code = raw_code
        if "```python" in raw_code:
            code = raw_code.split("```python")[1].split("```")[0].strip()
        elif "```" in raw_code:
            code = raw_code.split("```")[1].split("```")[0].strip()

        safe_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range,
                "str": str, "int": int, "float": float, "bool": bool,
                "list": list, "dict": dict, "set": set, "tuple": tuple,
                "zip": zip, "map": map, "filter": filter, "sum": sum,
                "min": min, "max": max, "abs": abs, "round": round,
                "sorted": sorted, "enumerate": enumerate, "reversed": reversed,
                "type": type, "isinstance": isinstance, "hasattr": hasattr,
                "getattr": getattr, "setattr": setattr,
                "Exception": Exception, "ValueError": ValueError,
                "KeyError": KeyError, "IndexError": IndexError,
                "ZeroDivisionError": ZeroDivisionError,
                "open": open, "input": lambda x="": "",
                "json": json, "os": os, "sys": sys,
                "math": __import__("math"),
                "random": __import__("random"),
                "datetime": __import__("datetime"),
                "re": __import__("re"),
                "itertools": __import__("itertools"),
                "collections": __import__("collections"),
                "statistics": __import__("statistics"),
                "hashlib": hashlib,
                "base64": base64,
                "io": io,
                "requests": requests,
                "numpy": np,
                "pandas": pd,
                "plt": plt,
                "matplotlib": matplotlib,
            },
            "np": np, "pd": pd, "plt": plt, "matplotlib": matplotlib,
            "BytesIO": BytesIO, "base64": base64,
        }

        output_buffer = io.StringIO()
        error_buffer = io.StringIO()

        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            try:
                exec(code, safe_globals)
            except Exception as e:
                print(f"Runtime Error: {type(e).__name__}: {str(e)}")

        output = output_buffer.getvalue()
        errors = error_buffer.getvalue()

        full_output = output if output.strip() else "Code executed successfully (no output)"
        if errors.strip():
            full_output += f"\n\nStderr:\n{errors}"

        return {
            "reply": f"""**Code Generated & Executed!**

```python
{code}
```

**Output:**
```
{full_output}
```""",
            "code": code,
            "output": full_output
        }
    except Exception as e:
        return {"reply": f"Code execution error: {str(e)}"}

# ========== 4. WEBSITE BUILDER ==========
@app.post("/build-website")
async def build_website(request: WebsiteRequest):
    try:
        client = get_client()

        website_prompt = f"""Create a complete, self-contained HTML website based on this request.

Request: {request.message}

Requirements:
- Create a single HTML file with embedded CSS and JavaScript
- Use modern, clean design with good UX
- Make it fully functional and interactive
- Include all necessary styles inline
- Use placeholder images from picsum.photos or similar if needed
- Return ONLY the complete HTML code, no explanations

Return the complete HTML code:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": website_prompt}],
            temperature=0.7,
            max_tokens=4096
        )

        html_code = response.choices[0].message.content

        # Extract HTML from markdown if present
        if "```html" in html_code:
            html_code = html_code.split("```html")[1].split("```")[0].strip()
        elif "```" in html_code:
            html_code = html_code.split("```")[1].split("```")[0].strip()

        # Save to file
        file_id = str(uuid.uuid4())[:8]
        filename = f"/tmp/website_{file_id}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_code)

        generated_files[request.session_id] = filename

        # Create a simple preview URL (in production, you'd host this)
        preview_url = f"/preview/{file_id}"

        return {
            "reply": f"Website created successfully! You can download or preview it below.",
            "code": html_code,
            "preview_url": preview_url,
            "file_id": file_id
        }
    except Exception as e:
        return {"reply": f"Website build error: {str(e)}"}

@app.get("/preview/{file_id}")
async def preview_website(file_id: str):
    for sid, path in generated_files.items():
        if file_id in path:
            return HTMLResponse(content=open(path, "r", encoding="utf-8").read())
    return {"error": "Website not found"}
# ========== 5. DATA ANALYSIS ==========
@app.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    try:
        client = get_client()

        analyze_prompt = f"""Analyze the following data/text and provide insights.

Data: {request.message}

Please provide:
1. Summary of the data
2. Key statistics and metrics
3. Trends or patterns
4. Actionable insights
5. Suggestions for visualization if applicable

Format your response with clear sections."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": analyze_prompt}],
            temperature=0.5,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        # Try to generate a chart if data looks like it has numbers
        chart_image = None
        try:
            # Simple heuristic: if message contains comma-separated numbers
            numbers = re.findall(r'\d+\.?\d*', request.message)
            if len(numbers) >= 3:
                chart_prompt = f"""Based on this data: {request.message}

Create a Python dictionary for a bar chart with this exact format:
{{"labels": ["label1", "label2", ...], "values": [val1, val2, ...], "title": "Chart Title"}}

Return ONLY the dictionary, no other text."""

                chart_response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": chart_prompt}],
                    temperature=0.3
                )

                chart_text = chart_response.choices[0].message.content
                # Extract dict
                dict_match = re.search(r'{.*}', chart_text, re.DOTALL)
                if dict_match:
                    chart_data = eval(dict_match.group())
                    chart_image = generate_chart("bar", chart_data, chart_data.get("title", "Analysis"))
        except:
            pass

        return {
            "reply": reply,
            "chart_image": chart_image,
            "mode": "analyze"
        }
    except Exception as e:
        return {"reply": f"Analysis error: {str(e)}"}

# ========== 6. FILE UPLOAD ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO documents (session_id, filename, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, file.filename, text, time.time())
        )
        conn.commit()
        conn.close()

        # Summarize with AI
        client = get_client()
        summary_prompt = f"""Summarize this document:

Filename: {file.filename}
Content (first 3000 chars): {text[:3000]}

Provide:
1. Brief summary
2. Key points
3. File type insights"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5
        )

        reply = response.choices[0].message.content

        return {
            "reply": reply,
            "filename": file.filename,
            "size": len(content),
            "text_preview": text[:500]
        }
    except Exception as e:
        return {"reply": f"Upload error: {str(e)}"}

# ========== 7. IMAGE GENERATION ==========
@app.post("/generate-image")
async def generate_image_endpoint(request: ImageRequest):
    try:
        prompt = request.message

        # Generate multiple variations
        images = []
        for i in range(3):
            img_url = generate_image(prompt, width=1024, height=768)
            images.append(img_url)

        return {
            "reply": f"Generated {len(images)} images for: '{prompt}'",
            "images": images,
            "mode": "image"
        }
    except Exception as e:
        return {"reply": f"Image generation error: {str(e)}"}

# ========== 8. WEB SEARCH ==========
@app.post("/search")
async def search_web(request: SearchRequest):
    try:
        results = web_search(request.message)
        return {
            "reply": results,
            "mode": "search"
        }
    except Exception as e:
        return {"reply": f"Search error: {str(e)}"}

# ========== 9. WEATHER ==========
@app.post("/weather")
async def weather_endpoint(request: WeatherRequest):
    try:
        # Extract city from message
        city = request.message.strip()
        if not city:
            city = "London"  # default

        result = get_weather(city)
        return {
            "reply": result,
            "mode": "weather"
        }
    except Exception as e:
        return {"reply": f"Weather error: {str(e)}"}

# ========== 10. NEWS ==========
@app.post("/news")
async def news_endpoint(request: NewsRequest):
    try:
        query = request.message.strip() or "technology"
        result = get_news(query)
        return {
            "reply": result,
            "mode": "news"
        }
    except Exception as e:
        return {"reply": f"News error: {str(e)}"}

# ========== 11. AGENT MODE ==========
@app.post("/agent")
async def agent_mode(request: AgentRequest):
    try:
        client = get_client()

        # First, let the LLM decide which tools to use
        tool_prompt = f"""You are an AI agent. Analyze this user request and decide which tools to use.

Available tools: search, code, weather, news, image, chart, translate, sentiment

User request: {request.message}

Respond with ONLY a JSON array of tool names to use, e.g.: ["search", "code"]
If no tools are needed, respond with: []"""

        tool_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": tool_prompt}],
            temperature=0.3
        )

        tool_text = tool_response.choices[0].message.content.strip()

        # Extract JSON array
        try:
            tools_to_use = json.loads(tool_text)
        except:
            # Fallback: extract from text
            tools_to_use = []
            for tool in ["search", "code", "weather", "news", "image", "chart", "translate", "sentiment"]:
                if tool in tool_text.lower():
                    tools_to_use.append(tool)

        # Execute tools
        tool_results = {}

        if "search" in tools_to_use:
            tool_results["search"] = web_search(request.message)

        if "weather" in tools_to_use:
            # Try to extract city
            cities = re.findall(r'in ([A-Za-z\s]+)', request.message)
            if cities:
                tool_results["weather"] = get_weather(cities[0])

        if "news" in tools_to_use:
            tool_results["news"] = get_news(request.message)

        if "code" in tools_to_use:
            # Generate code
            code_prompt = f"Write Python code for: {request.message}"
            code_response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": code_prompt}],
                temperature=0.3
            )
            tool_results["code"] = code_response.choices[0].message.content

        # Synthesize final response
        synthesis_prompt = f"""Synthesize a helpful response based on the user request and tool results.

User request: {request.message}

Tools used: {tools_to_use}

Tool results:
{json.dumps(tool_results, indent=2)}

Provide a comprehensive, well-structured response."""

        final_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": synthesis_prompt}],
            temperature=0.7,
            max_tokens=4096
        )

        reply = final_response.choices[0].message.content

        return {
            "reply": reply,
            "tools_used": tools_to_use,
            "mode": "agent"
        }
    except Exception as e:
        return {"reply": f"Agent error: {str(e)}"}

# ========== 12. TEXT TO SPEECH ==========
@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    try:
        audio = text_to_speech(request.message, request.lang)
        if audio:
            return {
                "reply": f"Text converted to speech: '{request.message[:100]}...'",
                "audio": audio,
                "mode": "tts"
            }
        else:
            return {
                "reply": "TTS not available. Install gTTS: pip install gtts",
                "mode": "tts"
            }
    except Exception as e:
        return {"reply": f"TTS error: {str(e)}"}

# ========== 13. CHART GENERATION ==========
@app.post("/chart")
async def chart_endpoint(request: ChartRequest):
    try:
        client = get_client()

        # Ask LLM to extract chart data from message
        chart_prompt = f"""Extract chart data from this request and return a Python dictionary.

Request: {request.message}

Return ONLY a dictionary in this exact format:
{{"chart_type": "bar|line|pie|scatter", "labels": [...], "values": [...], "title": "...", "xlabel": "...", "ylabel": "..."}}

If the request is vague, create sample data that makes sense."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": chart_prompt}],
            temperature=0.3
        )

        chart_text = response.choices[0].message.content

        # Extract dict
        dict_match = re.search(r'{.*}', chart_text, re.DOTALL)
        if dict_match:
            chart_data = eval(dict_match.group())
            chart_type = chart_data.get("chart_type", "bar")
            chart_image = generate_chart(chart_type, chart_data, chart_data.get("title", "Chart"))

            return {
                "reply": f"Generated {chart_type} chart: {chart_data.get('title', 'Chart')}",
                "chart_image": chart_image,
                "mode": "chart"
            }
        else:
            return {"reply": "Could not parse chart data from request."}
    except Exception as e:
        return {"reply": f"Chart error: {str(e)}"}

# ========== 14. YOUTUBE VIDEO CREATOR ==========
@app.post("/create-video")
async def create_video(request: VideoRequest):
    try:
        client = get_client()

        # Generate video script and metadata
        video_prompt = f"""Create a YouTube video package for this topic:

Topic: {request.message}

Provide:
1. Catchy video title
2. SEO-optimized description (200 words)
3. 10 relevant tags
4. Video script outline (intro, 3 main points, outro with CTA)
5. Thumbnail text suggestion
6. Target audience

Format clearly with headers."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": video_prompt}],
            temperature=0.7,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        # Generate thumbnail
        thumbnail_url = generate_image(f"YouTube thumbnail: {request.message}, bold text, eye-catching, professional, 16:9 aspect ratio", width=1280, height=720)

        return {
            "reply": reply,
            "thumbnail_url": thumbnail_url,
            "mode": "video"
        }
    except Exception as e:
        return {"reply": f"Video creation error: {str(e)}"}

# ========== 15. TRANSLATION ==========
@app.post("/translate")
async def translate_endpoint(request: TranslateRequest):
    try:
        client = get_client()

        translate_prompt = f"""Translate the following text to {request.target_language}.

Text: {request.message}

Provide:
1. The translation
2. Pronunciation guide (if applicable)
3. Cultural notes or context
4. Alternative translations if ambiguous"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": translate_prompt}],
            temperature=0.3
        )

        reply = response.choices[0].message.content

        return {
            "reply": reply,
            "source_language": "auto-detected",
            "target_language": request.target_language,
            "mode": "translate"
        }
    except Exception as e:
        return {"reply": f"Translation error: {str(e)}"}

# ========== 16. SENTIMENT ANALYSIS ==========
@app.post("/sentiment")
async def sentiment_endpoint(request: SentimentRequest):
    try:
        client = get_client()

        sentiment_prompt = f"""Analyze the sentiment of this text comprehensively.

Text: {request.message}

Provide:
1. Overall sentiment (Positive/Negative/Neutral/Mixed) with confidence score
2. Emotion breakdown (joy, anger, sadness, fear, surprise, disgust) with percentages
3. Key phrases that indicate sentiment
4. Tone analysis (formal, casual, aggressive, supportive, etc.)
5. Actionable insights if this is customer feedback

Format as a structured report."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": sentiment_prompt}],
            temperature=0.3
        )

        reply = response.choices[0].message.content

        return {
            "reply": reply,
            "mode": "sentiment"
        }
    except Exception as e:
        return {"reply": f"Sentiment analysis error: {str(e)}"}

# ========== 17. RAG (DOCUMENT QA) ==========
@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)

        # Store in RAG documents
        if session_id not in rag_documents:
            rag_documents[session_id] = []

        rag_documents[session_id].append({
            "filename": file.filename,
            "content": text,
            "timestamp": time.time()
        })

        # Also save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO documents (session_id, filename, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, file.filename, text, time.time())
        )
        conn.commit()
        conn.close()

        return {
            "reply": f"Document '{file.filename}' uploaded successfully. You can now ask questions about it in RAG mode.",
            "filename": file.filename,
            "chunks": len(text) // 500  # rough chunk count
        }
    except Exception as e:
        return {"reply": f"RAG upload error: {str(e)}"}

@app.post("/rag/query")
async def rag_query(request: RAGQueryRequest):
    try:
        session_id = request.session_id or "default"

        # Get all documents for this session
        docs = rag_documents.get(session_id, [])

        if not docs:
            # Try DB
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute(
                "SELECT filename, content FROM documents WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,)
            )
            rows = c.fetchall()
            conn.close()
            docs = [{"filename": r[0], "content": r[1]} for r in rows]

        if not docs:
            return {"reply": "No documents found. Please upload documents first in RAG mode."}

        # Combine all document content
        all_content = "\n\n".join([f"--- {d['filename']} ---\n{d['content'][:5000]}" for d in docs])

        client = get_client()

        rag_prompt = f"""Answer the following question based ONLY on the provided documents.

Documents:
{all_content[:15000]}

Question: {request.message}

Instructions:
- Answer based only on the documents provided
- If the answer is not in the documents, say so clearly
- Cite which document(s) you used
- Be concise but thorough"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": rag_prompt}],
            temperature=0.3,
            max_tokens=4096
        )

        reply = response.choices[0].message.content

        return {
            "reply": reply,
            "documents_used": [d["filename"] for d in docs],
            "mode": "rag"
        }
    except Exception as e:
        return {"reply": f"RAG query error: {str(e)}"}

# ========== 18. CLEAR HISTORY ==========
@app.post("/clear")
async def clear_history(request: ClearRequest):
    session_id = request.session_id or "default"

    if session_id in conversations:
        conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPTS["chat"]}
        ]

    # Clear from DB too
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

    return {"status": "cleared", "session_id": session_id}

# ========== 19. GET HISTORY ==========
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = get_conversation_history(session_id, limit=50)
    return {"history": history, "session_id": session_id}
# ========== 20. HEALTH CHECK ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "features_active": {
            "groq": bool(GROQ_API_KEY),
            "web_search": bool(TAVILY_API_KEY),
            "weather": bool(OPENWEATHER_API_KEY),
            "news": bool(NEWSDATA_API_KEY),
            "tts": GTTS_AVAILABLE,
            "pdf": PYPDF_AVAILABLE,
            "excel": OPENPYXL_AVAILABLE,
            "faiss": FAISS_AVAILABLE
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
