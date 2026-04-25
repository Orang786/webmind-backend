import os
import json
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def build_messages(query: str, history: list, sites_data: list = None) -> tuple:
    """Собираем сообщения для AI"""
    
    context_str = ""
    if sites_data:
        context_str = "\n\n📌 Данные из интернета:\n"
        for site in sites_data:
            if site.get("content"):
                context_str += f"\n[{site['title']}]({site['url']})\n{site['content'][:3000]}\n"

    system_prompt = """Ты — WebMind AI, умный помощник.

Правила:
1. Отвечай ТОЛЬКО на русском языке.
2. Используй Markdown для форматирования:
   - **жирный** для важного
   - `код` для технических терминов
   - Блоки кода с указанием языка
   - ## Заголовки для структуры
   - - списки для перечислений
   - | таблицы | для сравнений |
3. Если есть данные из интернета — анализируй их.
4. Отвечай развёрнуто и структурированно."""

    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    user_content = f"{context_str}\n\nВопрос: {query}" if context_str else query
    messages.append({"role": "user", "content": user_content})

    return system_prompt, messages


async def stream_analyze(query: str, history: list, sites_data: list = None):
    """Стриминг ответа — генерирует чанки текста"""
    
    system_prompt, messages = build_messages(query, history, sites_data)

    try:
        stream = await client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            stream=True  # Включаем стриминг!
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    except Exception as e:
        yield f"\n\n❌ Ошибка: {str(e)}"


async def decide_and_analyze(query: str, history: list, sites_data: list = None) -> dict:
    """Обычный запрос без стриминга"""
    
    system_prompt, messages = build_messages(query, history, sites_data)

    try:
        response = await client.chat.completions.create(
            model="google/gemma-3-27b-it:free",
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
        )

        answer = response.choices[0].message.content or ""

        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = response.usage.total_tokens or 0

        return {
            "answer": answer,
            "tokens_used": tokens_used
        }

    except Exception as e:
        print(f"❌ Ошибка AI: {e}")
        return {
            "answer": f"Произошла ошибка: {str(e)}",
            "tokens_used": 0
        }