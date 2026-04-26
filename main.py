from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from services.search import search_sites
from services.scraper import scrape_sites
from services.ai_analyzer import stream_analyze, decide_and_analyze
from models.schemas import SearchRequest, SearchResponse
import json
import os

app = FastAPI(title="WebMind AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "WebMind AI is running!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/test-env")
async def test_env():
    key = os.getenv("OPENROUTER_API_KEY", "НЕ НАЙДЕН")
    return {
        "key_found": key != "НЕ НАЙДЕН",
        "key_preview": key[:15] + "..." if len(key) > 15 else key
    }

@app.post("/stream")
async def stream_endpoint(request: SearchRequest):
    """Стриминг эндпоинт — возвращает текст по частям"""
    
    skip_search_phrases = [
        "привет", "как дела", "кто ты", "спасибо",
        "ясно", "понятно", "окей", "ок", "пока", "хорошо"
    ]
    query_lower = request.query.lower().strip()
    need_search = (
        len(request.query) > 12 and
        not any(p in query_lower for p in skip_search_phrases)
    )

    sites = []
    sources = []

    if need_search:
        print(f"🔍 Поиск: {request.query}")
        search_results = await search_sites(request.query)
        sites = await scrape_sites(search_results)
        sources = [
            {"url": s["url"], "title": s["title"], "snippet": s["snippet"]}
            for s in sites
        ]
        print(f"✅ Найдено {len(sites)} сайтов")

    history_list = [m.dict() for m in request.history]

    async def generate():
        # Сначала отправляем метаданные (источники, флаг поиска)
        meta = {
            "type": "meta",
            "sources": sources,
            "is_search_performed": need_search
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # Потом стримим текст
        print("🧠 AI стримит...")
        async for chunk in stream_analyze(request.query, history_list, sites if need_search else None):
            payload = {"type": "chunk", "text": chunk}
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        # Сигнал конца
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        print("✅ Стриминг завершён")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/analyze", response_model=SearchResponse)
async def analyze(request: SearchRequest):
    try:
        skip_search_phrases = [
            "привет", "как дела", "кто ты", "спасибо",
            "ясно", "понятно", "окей", "ок", "пока"
        ]
        query_lower = request.query.lower().strip()
        need_search = (
            len(request.query) > 12 and
            not any(p in query_lower for p in skip_search_phrases)
        )

        sites = []
        if need_search:
            print(f"🔍 Поиск: {request.query}")
            search_results = await search_sites(request.query)
            sites = await scrape_sites(search_results)

        history_list = [m.dict() for m in request.history]
        ai_result = await decide_and_analyze(
            query=request.query,
            history=history_list,
            sites_data=sites if need_search else None
        )

        clean_sources = [
            {"url": s["url"], "title": s["title"], "snippet": s["snippet"]}
            for s in sites
        ]

        return SearchResponse(
            answer=ai_result.get("answer", ""),
            sources=clean_sources,
            is_search_performed=need_search,
            tokens_used=ai_result.get("tokens_used", 0)
        )

    except Exception as e:
        print(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/models")
async def list_models():
    """Показывает доступные модели на OpenRouter"""
    import httpx
    import os
    
    async with httpx.AsyncClient() as client:
        res = await client.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}"}
        )
    
    models = res.json()
    # Фильтруем только бесплатные и Google модели
    filtered = [
        m["id"] for m in models.get("data", [])
        if "gemini" in m["id"].lower() or ":free" in m["id"]
    ]
    return {"available_models": filtered}