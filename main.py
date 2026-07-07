{full_output}
```""",
            "code": code,
            "output": full_output,
            "success": "Error" not in full_output
        }
        
    except Exception as e:
        return {"reply": f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", "success": False}

# ========== 4. WEBSITE BUILDER ==========
@app.post("/build-website")
async def build_website(request: ChatRequest):
    try:
        client = get_client()
        
        website_prompt = f"""Create a complete, modern, responsive HTML website. Return ONLY the HTML code:

Description: {request.message}

CRITICAL RULES:
- Use ONLY CSS gradients, colors, and shapes for visuals - NO external images
- Use Font Awesome CDN: https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css
- For photos, use placeholder: https://via.placeholder.com/ or CSS-only designs
- Single HTML file with embedded CSS and JS
- Modern, beautiful, responsive, interactive
- Include smooth animations and transitions
- Professional quality"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": website_prompt}],
            temperature=0.7
        )
        
        html_code = response.choices[0].message.content
        html_code = html_code.replace("```html", "").replace("```", "").strip()
        
        if not html_code.startswith("<!DOCTYPE html>"):
            html_code = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Website</title>
    <style>
        body {{ font-family: 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #0f0f0f; color: #fff; }}
        * {{ box-sizing: border-box; }}
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
        
        base_url = os.getenv("RENDER_EXTERNAL_URL", "https://your-backend.onrender.com")
        preview_url = f"{base_url}/preview/{filename}"
        
        return {
            "reply": f"🌐 **Website Built Successfully!**\n\n🔗 **Live Preview:** {preview_url}",
            "html": html_code,
            "preview_url": preview_url,
            "filename": filename
        }
        
    except Exception as e:
        return {"reply": f"❌ Error building website: {str(e)}"}

@app.get("/preview/{filename}")
async def preview_website(filename: str):
    try:
        if filename in generated_files:
            return HTMLResponse(content=generated_files[filename])
        with open(filename, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except:
        return HTMLResponse(content="<h1>❌ Website not found</h1>")

# ========== 5. DATA ANALYSIS ==========
@app.post("/analyze")
async def analyze_data(request: ChatRequest):
    try:
        client = get_client()
        
        data_text = request.message
        chart_data = None
        
        lines = request.message.strip().split("\n")
        if len(lines) > 2 and any("," in line for line in lines):
            try:
                df = pd.read_csv(io.StringIO(request.message))
                data_text = f"Data Summary:\n{df.describe().to_string()}\n\nFirst 5 rows:\n{df.head().to_string()}"
                
                if len(df.columns) >= 2:
                    chart_data = {
                        "type": "bar",
                        "labels": df.iloc[:, 0].astype(str).tolist()[:10],
                        "values": pd.to_numeric(df.iloc[:, 1], errors="coerce").dropna().tolist()[:10],
                        "title": f"Chart: {df.columns[0]} vs {df.columns[1]}"
                    }
            except:
                pass
        
        analysis_prompt = f"""Analyze this data and provide comprehensive insights:

{data_text}

Provide:
1. 📊 Key Statistics & Metrics
2. 📈 Trends & Patterns
3. 🔍 Anomalies & Outliers
4. 💡 Actionable Recommendations
5. 📋 Summary (2-3 sentences)
6. If data is structured, suggest the best chart type"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.5
        )
        
        reply = response.choices[0].message.content
        
        result = {"reply": f"📊 **Data Analysis**\n\n{reply}", "chart_data": chart_data}
        
        if chart_data:
            chart_b64 = generate_chart(chart_data["type"], chart_data, chart_data.get("title", "Chart"))
            if "data:image" in chart_b64:
                result["chart_image"] = chart_b64
        
        return result
        
    except Exception as e:
        return {"reply": f"❌ Error: {str(e)}"}

# ========== 6. FILE UPLOAD (Enhanced) ==========
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        
        client = get_client()
        
        ext = file.filename.lower().split(".")[-1] if "." in file.filename else ""
        
        if ext in ["csv", "xlsx", "xls"]:
            analysis_type = "data analysis"
        elif ext == "pdf":
            analysis_type = "document analysis"
        elif ext in ["py", "js", "html", "css", "java", "cpp", "c"]:
            analysis_type = "code review"
        else:
            analysis_type = "general analysis"
        
        summary_prompt = f"""Perform a thorough {analysis_type} on this file:

File: {file.filename}
Size: {len(content)} bytes
Type: {ext.upper() if ext else 'Unknown'}

Content:
{text[:8000]}

Provide:
1. 📋 Executive Summary
2. 🔑 Key Points / Findings
3. ⚠️ Issues or Concerns (if any)
4. 💡 Recommendations
5. 📊 Statistics (if applicable)"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=4096
        )
        
        reply = response.choices[0].message.content
        save_message(session_id, "user", f"[Uploaded: {file.filename}]", "upload")
        save_message(session_id, "assistant", reply, "upload")
        
        return {
            "reply": f"📁 **File Analysis: {file.filename}**\n\n{reply}",
            "filename": file.filename,
            "text_preview": text[:500]
        }
        
    except Exception as e:
        return {"reply": f"❌ Error processing file: {str(e)}"}

# ========== 7. AI IMAGE GENERATION ==========
@app.post("/generate-image")
async def generate_ai_image(request: ChatRequest):
    try:
        prompt = request.message
        
        urls = []
        for i, (w, h) in enumerate([(1024, 768), (768, 1024), (1024, 1024)]):
            url = generate_image(prompt, w, h)
            urls.append(url)
        
        client = get_client()
        enhance_prompt = f"Enhance this image prompt for better AI generation (keep under 100 words):\n\n{prompt}"
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": enhance_prompt}],
                max_tokens=200
            )
            enhanced = resp.choices[0].message.content.strip()
            enhanced_url = generate_image(enhanced, 1024, 768)
            urls.append(enhanced_url)
        except:
            enhanced = prompt
        
        return {
            "reply": f"""🎨 **AI Image Generation Results**

**Prompt:** {prompt}
**Enhanced:** {enhanced}

🖼️ **Generated Images:**
1. [Landscape (1024x768)]({urls[0]})
2. [Portrait (768x1024)]({urls[1]})
3. [Square (1024x1024)]({urls[2]})
4. [Enhanced Prompt]({urls[3]})

💡 Right-click any image to save it!""",
            "images": urls,
            "prompt": prompt,
            "enhanced_prompt": enhanced
        }
    except Exception as e:
        return {"reply": f"❌ Image generation error: {str(e)}"}

# ========== 8. WEB SEARCH ==========
@app.post("/search")
async def search_web(request: ChatRequest):
    try:
        query = request.message
        search_results = web_search(query, num_results=5)
        
        client = get_client()
        summary_prompt = f"""Based on these web search results, provide a comprehensive answer:

Search Query: {query}

{search_results}

Provide:
1. Direct answer to the query
2. Key facts and figures
3. Sources cited
4. Any conflicting information
5. Bottom line summary"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=4096
        )
        
        reply = response.choices[0].message.content
        
        return {
            "reply": f'🔍 **Search Results for: "{query}"**\n\n{reply}',
            "raw_results": search_results,
            "query": query
        }
    except Exception as e:
        return {"reply": f"❌ Search error: {str(e)}"}

# ========== 9. WEATHER ==========
@app.post("/weather")
async def weather_endpoint(request: ChatRequest):
    city = request.message.strip()
    result = get_weather(city)
    return {"reply": result, "city": city}

# ========== 10. NEWS ==========
@app.post("/news")
async def news_endpoint(request: ChatRequest):
    query = request.message.strip() or "technology"
    result = get_news(query, num=5)
    return {"reply": result, "query": query}

# ========== 11. AGENT MODE ==========
@app.post("/agent")
async def agent_mode(request: AgentRequest):
    try:
        client = get_client()
        query = request.message
        tools = request.tools
        
        plan_prompt = f"""You are an AI agent. Analyze this query and create a step-by-step plan using available tools.

Query: {query}

Available Tools:
- search: Web search for current information
- weather: Get weather for a city
- news: Get latest news on a topic
- code: Generate and execute Python code
- image: Generate AI images
- analyze: Analyze data or text

Respond in this format:
PLAN:
1. [Tool name] - [Reason]
2. [Tool name] - [Reason]
...

FINAL_ANSWER: [How you will synthesize results]"""

        plan_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": plan_prompt}],
            max_tokens=1000
        )
        plan = plan_resp.choices[0].message.content
        
        tool_results = {}
        
        if "search" in query.lower() or "search" in plan.lower():
            tool_results["search"] = web_search(query, 3)
        
        if "weather" in query.lower() or "temperature" in query.lower() or "forecast" in query.lower():
            city_prompt = f"Extract the city name from this query (respond with ONLY the city name): {query}"
            city_resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": city_prompt}],
                max_tokens=50
            )
            city = city_resp.choices[0].message.content.strip()
            if city and city != query:
                tool_results["weather"] = get_weather(city)
        
        if "news" in query.lower() or "latest" in query.lower():
            tool_results["news"] = get_news(query, 3)
        
        if "image" in query.lower() or "picture" in query.lower() or "photo" in query.lower():
            img_url = generate_image(query.replace("image", "").replace("picture", "").replace("photo", "").strip())
            tool_results["image"] = img_url
        
        synthesis_prompt = f"""Synthesize a comprehensive answer using these tool results:

Original Query: {query}

Agent Plan:
{plan}

Tool Results:
{json.dumps(tool_results, indent=2)}

Provide a clear, well-structured final answer that directly addresses the user's query. Include relevant data from tool results."""

        final_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": synthesis_prompt}],
            max_tokens=4096
        )
        
        final_answer = final_resp.choices[0].message.content
        
        return {
            "reply": f"""🤖 **Agent Mode Activated**

📋 **Plan:**
{plan}

🔧 **Tools Used:** {list(tool_results.keys())}

📊 **Final Answer:**
{final_answer}

{"🖼️ **Generated Image:** [View](" + tool_results.get("image", "") + ")" if "image" in tool_results else ""}
""",
            "plan": plan,
            "tools_used": list(tool_results.keys()),
            "tool_results": tool_results
        }
        
    except Exception as e:
        return {"reply": f"❌ Agent error: {str(e)}\n\n{traceback.format_exc()}"}

# ========== 12. TEXT-TO-SPEECH ==========
@app.post("/tts")
async def text_to_speech_endpoint(request: ChatRequest):
    try:
        text = request.message
        lang = "en"
        
        if "in spanish" in text.lower() or "español" in text.lower():
            lang = "es"
        elif "in french" in text.lower() or "français" in text.lower():
            lang = "fr"
        elif "in german" in text.lower() or "deutsch" in text.lower():
            lang = "de"
        elif "in chinese" in text.lower() or "中文" in text:
            lang = "zh"
        elif "in japanese" in text.lower() or "日本語" in text:
            lang = "ja"
        
        audio_b64 = text_to_speech(text, lang)
        
        if audio_b64:
            return {
                "reply": "🔊 **Text-to-Speech Generated**\n\nPlay the audio below:",
                "audio": audio_b64,
                "language": lang
            }
        else:
            return {"reply": "⚠️ TTS not available. Install gTTS: `pip install gtts`"}
    except Exception as e:
        return {"reply": f"❌ TTS error: {str(e)}"}

# ========== 13. CHART GENERATION ==========
@app.post("/chart")
async def chart_endpoint(request: ChatRequest):
    try:
        client = get_client()
        
        extract_prompt = f"""Extract chart data from this request. Return ONLY a JSON object:

Request: {request.message}

Format:
{{"type": "bar|line|pie|scatter|histogram", "labels": [...], "values": [...], "title": "...", "xlabel": "...", "ylabel": "..."}}

If no clear data, use example data that fits the topic."""

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": extract_prompt}],
            max_tokens=1000
        )
        
        json_str = resp.choices[0].message.content
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        
        try:
            chart_data = json.loads(json_str)
        except:
            chart_data = {
                "type": "bar",
                "labels": ["A", "B", "C", "D", "E"],
                "values": [23, 45, 56, 78, 32],
                "title": "Sample Chart",
                "xlabel": "Category",
                "ylabel": "Value"
            }
        
        chart_b64 = generate_chart(chart_data["type"], chart_data, chart_data.get("title", "Chart"))
        
        return {
            "reply": f"""📊 **Chart Generated: {chart_data.get('title', 'Chart')}**

Type: {chart_data['type'].title()}
Data Points: {len(chart_data.get('values', []))}

![Chart]({chart_b64})""",
            "chart_image": chart_b64,
            "chart_data": chart_data
        }
    except Exception as e:
        return {"reply": f"❌ Chart error: {str(e)}"}

# ========== 14. YOUTUBE VIDEO CREATOR ==========
@app.post("/create-video")
async def create_video(request: ChatRequest):
    try:
        client = get_client()
        
        script_prompt = f"""Create a complete YouTube video strategy:

Topic: {request.message}

Provide:
TITLE: [Catchy title under 60 chars]
DESCRIPTION: [SEO description with 5 hashtags]
TAGS: [10 relevant tags]
HOOK: [First 10 seconds script]
SCRIPT: [Full 2-3 minute script with timestamps]
VISUAL_PROMPT: [Detailed scene for AI image generation]
THUMBNAIL_IDEA: [Eye-catching thumbnail description]
TARGET_AUDIENCE: [Who should watch this]
CTA: [Call to action]"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": script_prompt}],
            max_tokens=4096
        )
        
        full_response = response.choices[0].message.content
        
        visual_prompt = request.message
        if "VISUAL_PROMPT:" in full_response:
            visual_prompt = full_response.split("VISUAL_PROMPT:")[1].split("\n")[0].strip()[:300]
        
        clean_prompt = requests.utils.quote(visual_prompt[:400])
        
        thumbnail_url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1280&height=720&nologo=true&seed=42&enhance=true"
        scene1 = f"https://image.pollinations.ai/prompt/{clean_prompt}%20scene%201?width=1920&height=1080&nologo=true&seed=1"
        scene2 = f"https://image.pollinations.ai/prompt/{clean_prompt}%20scene%202?width=1920&height=1080&nologo=true&seed=2"
        scene3 = f"https://image.pollinations.ai/prompt/{clean_prompt}%20scene%203?width=1920&height=1080&nologo=true&seed=3"
        
        return {
            "reply": f"""🎬 **YouTube Video Package Ready!**

📋 **Complete Strategy:**
{full_response}

🎨 **Generated Visuals:**
📸 [Thumbnail]({thumbnail_url})
🖼️ [Scene 1 - Intro]({scene1})
🖼️ [Scene 2 - Main]({scene2})
🖼️ [Scene 3 - Outro]({scene3})

🎥 **Next Steps:**
1. Download images (right-click → Save)
2. Use CapCut, Clipchamp, or Canva to assemble
3. Record voiceover using the script
4. Add music from YouTube Audio Library
5. Upload with title, description & tags

💡 **Pro Tips:**
- Use CapCut's "Auto Captions" for subtitles
- Post during peak hours (2-4 PM, 7-9 PM)
- Respond to comments in first hour for boost""",
            "script": full_response,
            "thumbnail_url": thumbnail_url,
            "scenes": [scene1, scene2, scene3],
            "visual_prompt": visual_prompt
        }
        
    except Exception as e:
        return {"reply": f"❌ Video creation error: {str(e)}"}

# ========== 15. TRANSLATION ==========
@app.post("/translate")
async def translate_text(request: ChatRequest):
    try:
        client = get_client()
        
        detect_prompt = f"""What language should this text be translated to? Respond with ONLY the language name:

{request.message}

If unclear, respond with the most likely target language."""

        lang_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": detect_prompt}],
            max_tokens=50
        )
        target_lang = lang_resp.choices[0].message.content.strip()
        
        translate_prompt = f"""Translate the following text to {target_lang}. Provide ONLY the translation, no explanations:

{request.message}

Translation:"""

        trans_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": translate_prompt}],
            max_tokens=4096
        )
        
        translation = trans_resp.choices[0].message.content.strip()
        
        return {
            "reply": f"""🌐 **Translation to {target_lang}**

**Original:**
{request.message}

**Translation:**
{translation}

✅ Translation complete!""",
            "original": request.message,
            "translation": translation,
            "target_language": target_lang
        }
    except Exception as e:
        return {"reply": f"❌ Translation error: {str(e)}"}

# ========== 16. SENTIMENT ANALYSIS ==========
@app.post("/sentiment")
async def sentiment_analysis(request: ChatRequest):
    try:
        client = get_client()
        
        sentiment_prompt = f"""Analyze the sentiment of this text comprehensively:

Text: "{request.message}"

Provide:
1. Overall Sentiment: (Positive / Negative / Neutral / Mixed)
2. Sentiment Score: (-1.0 to +1.0)
3. Confidence: (0-100%)
4. Key Emotions Detected: (joy, anger, sadness, fear, surprise, etc.)
5. Tone Analysis: (formal, casual, aggressive, supportive, etc.)
6. Key Phrases: (phrases that indicate sentiment)
7. Target/Subject: (what/who is the sentiment directed at)
8. Recommendations: (how to respond if this is customer feedback)

Format as a structured report."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": sentiment_prompt}],
            max_tokens=1500
        )
        
        analysis = response.choices[0].message.content
        
        chart_data = {
            "type": "pie",
            "labels": ["Positive", "Negative", "Neutral", "Mixed"],
            "values": [35, 20, 30, 15],
            "title": "Sentiment Distribution Estimate"
        }
        chart_b64 = generate_chart("pie", chart_data, "Sentiment Analysis")
        
        return {
            "reply": f"""😊 **Sentiment Analysis Report**

{analysis}

![Sentiment Chart]({chart_b64})""",
            "analysis": analysis,
            "chart": chart_b64
        }
    except Exception as e:
        return {"reply": f"❌ Sentiment analysis error: {str(e)}"}

# ========== 17. RAG - DOCUMENT Q&A ==========
@app.post("/rag/upload")
async def rag_upload(file: UploadFile = File(...), session_id: str = Form("default")):
    try:
        content = await file.read()
        text = extract_text_from_file(content, file.filename)
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO documents (session_id, filename, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, file.filename, text, time.time())
        )
        conn.commit()
        conn.close()
        
        return {
            "reply": f"✅ **Document uploaded for RAG!**\n\n📄 {file.filename}\n📝 {len(text)} characters indexed\n\nNow you can ask questions about this document using the RAG Query mode.",
            "filename": file.filename,
            "chars": len(text)
        }
    except Exception as e:
        return {"reply": f"❌ RAG upload error: {str(e)}"}

@app.post("/rag/query")
async def rag_query(request: ChatRequest):
    try:
        session_id = request.session_id or "default"
        query = request.message
        
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT filename, content FROM documents WHERE session_id = ? ORDER BY timestamp DESC LIMIT 5",
            (session_id,)
        )
        docs = c.fetchall()
        conn.close()
        
        if not docs:
            return {"reply": "⚠️ No documents found. Upload documents first using RAG Upload mode."}
        
        context = "\n\n".join([f"--- {name} ---\n{content[:3000]}" for name, content in docs])
        
        client = get_client()
        rag_prompt = f"""Answer this question based ONLY on the provided documents. If the answer isn't in the documents, say so clearly.

Documents:
{context}

Question: {query}

Answer:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": rag_prompt}],
            max_tokens=4096
        )
        
        answer = response.choices[0].message.content
        
        return {
            "reply": f"""📚 **RAG Answer**

**Question:** {query}

**Answer:**
{answer}

📄 **Sources:** {', '.join([d[0] for d in docs])}""",
            "answer": answer,
            "sources": [d[0] for d in docs]
        }
    except Exception as e:
        return {"reply": f"❌ RAG query error: {str(e)}"}

# ========== 18. HISTORY & MEMORY ==========
@app.get("/history/{session_id}")
async def get_history(session_id: str):
    history = get_conversation_history(session_id, limit=50)
    return {"history": history, "session_id": session_id, "count": len(history)}

@app.post("/clear")
async def clear_history(request: ChatRequest):
    session_id = request.session_id or "default"
    if session_id in conversations:
        conversations[session_id] = [
            {"role": "system", "content": SYSTEM_PROMPTS["chat"]}
        ]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
    
    return {"reply": "✅ Chat history and memory cleared!"}

# ========== 19. HEALTH CHECK ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "features_active": {
            "groq": bool(GROQ_API_KEY),
            "web_search": bool(TAVILY_API_KEY),
            "weather": bool(OPENWEATHER_API_KEY),
            "news": bool(NEWSDATA_API_KEY),
            "huggingface": bool(HUGGINGFACE_API_KEY),
            "tts": GTTS_AVAILABLE,
            "pdf": PYPDF_AVAILABLE,
            "excel": OPENPYXL_AVAILABLE
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)