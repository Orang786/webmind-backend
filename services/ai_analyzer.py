import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Модели по приоритету
MODELS = [
    "google/gemini-2.0-flash-001",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-4-31B:free",
]

def build_messages(query: str, history: list, sites_data: list = None) -> tuple:
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


async def try_models(messages: list, stream: bool = False):
    """Пробуем модели по очереди"""
    last_error = None
    for model in MODELS:
        try:
            print(f"🤖 Пробуем: {model}")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream
            )
            print(f"✅ Работает: {model}")
            return response
        except Exception as e:
            print(f"❌ {model}: {e}")
            last_error = e
            continue
    raise Exception(f"Все модели недоступны: {last_error}")


async def stream_analyze(query: str, history: list, sites_data: list = None):
    system_prompt, messages = build_messages(query, history, sites_data)
    all_messages = [
        {"role": "system", "content": system_prompt},
        *messages
    ]
    try:
        stream = await try_models(all_messages, stream=True)
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
    except Exception as e:
        yield f"\n\n❌ Ошибка: {str(e)}"


async def decide_and_analyze(query: str, history: list, sites_data: list = None) -> dict:
    system_prompt, messages = build_messages(query, history, sites_data)
    all_messages = [
        {"role": "system", "content": system_prompt},
        *messages
    ]
    try:
        response = await try_models(all_messages, stream=False)
        answer = response.choices[0].message.content or ""
        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = response.usage.total_tokens or 0
        return {"answer": answer, "tokens_used": tokens_used}
    except Exception as e:
        return {
            "answer": "Все AI модели сейчас перегружены. Попробуйте через минуту.",
            "tokens_used": 0
        }