import httpx
from bs4 import BeautifulSoup
import asyncio

# Сайты которые блокируют парсинг
BLOCKED_DOMAINS = [
    "facebook.com", "instagram.com", "twitter.com",
    "x.com", "tiktok.com", "linkedin.com", "vk.com"
]

async def scrape_one_site(url: str) -> str:
    # Проверяем заблокированные домены
    if any(domain in url for domain in BLOCKED_DOMAINS):
        print(f"⛔ Пропускаем заблокированный сайт: {url}")
        return ""

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
                follow_redirects=True
            )

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text[:3000]

    except Exception as e:
        print(f"Ошибка парсинга {url}: {e}")
        return ""

async def scrape_sites(sites: list) -> list:
    tasks = [scrape_one_site(site["url"]) for site in sites]
    contents = await asyncio.gather(*tasks)

    result = []
    for site, content in zip(sites, contents):
        result.append({
            **site,
            "content": content
        })

    return result