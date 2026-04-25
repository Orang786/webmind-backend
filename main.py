from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.search import search_sites
from services.scraper import scrape_sites
from services.ai_analyzer import decide_and_analyze
from models.schemas import SearchRequest, SearchResponse

app = FastAPI(title="WebMind AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze", response_model=SearchResponse)
async def analyze(request: SearchRequest):
    try:
        # Простые фразы — не ищем
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
            print(f"✅ Найдено {len(sites)} сайтов")

        history_list = [m.dict() for m in request.history]

        print("🧠 AI думает...")
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