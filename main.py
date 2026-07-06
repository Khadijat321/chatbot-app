from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import os

app = FastAPI()

# Allow Vercel frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# Replace with your actual Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

@app.post("/chat")
async def chat(request: ChatRequest):
    if request.session_id not in conversations:
        conversations[request.session_id] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    
    conversations[request.session_id].append(
        {"role": "user", "content": request.message}
    )
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=conversations[request.session_id]
        )
        reply = response.choices[0].message.content
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}
    
    conversations[request.session_id].append(
        {"role": "assistant", "content": reply}
    )
    
    return {"reply": reply}

@app.get("/")
async def root():
    return {"message": "Chatbot backend is running!"}