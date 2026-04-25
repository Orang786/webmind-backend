import httpx
import os
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

async def search_sites(query: str) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={
                "q": query,
                "num": 5,
                "hl": "ru"
            }
        )

    data = response.json()

    sites = []
    for item in data.get("organic", []):
        sites.append({
            "url": item.get("link"),
            "title": item.get("title"),
            "snippet": item.get("snippet", "")
        })

    return sites