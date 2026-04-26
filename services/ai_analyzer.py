import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Только проверенные рабочие модели
MODELS = [
    "google/gemini-2.5-flash",                    # Самая умная и быстрая
    "google/gemini-2.0-flash-001",                # Проверенная рабочая
    "google/gemini-2.0-flash-lite-001",           # Лёгкая версия
    "meta-llama/llama-3.3-70b-instruct:free",     # Бесплатная большая
    "google/gemma-3-27b-it:free",                 # Бесплатная от Google
    "meta-llama/llama-3.2-3b-instruct:free",      # Бесплатная маленькая
]

SYSTEM_PROMPT = """Ты — WebMind AI, умный помощник. 
Отвечай ТОЛЬКО на русском языке.
Используй Markdown для форматирования:
- **жирный** для важного
- ## Заголовки для структуры  
- - списки для перечислений
- | таблицы | для сравнений |
- ```язык для блоков кода```
Отвечай развёрнуто и структурированно."""


def build_messages(query: str, history: list, sites_data: list = None) -> list:
    """Собираем сообщения БЕЗ system роли — она идёт в первый user"""
    
    context_str = ""
    if sites_data:
        context_str = "\n\n📌 Данные из интернета:\n"
        for site in sites_data:
            if site.get("content"):
                context_str += f"\n[{site['title']}]\n{site['content'][:2000]}\n"

    messages = []
    
    # История предыдущих сообщений
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Текущий запрос пользователя
    # Инструкции добавляем прямо сюда чтобы не было проблем с system
    user_content = f"{SYSTEM_PROMPT}\n\n{context_str}\n\nВопрос: {query}"
    messages.append({"role": "user", "content": user_content})

    return messages


async def try_models(messages: list, stream: bool = False):
    """Пробуем модели по очереди пока одна не сработает"""
    last_error = None

    for model in MODELS:
        try:
            print(f"🤖 Пробуем: {model}")
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                max_tokens=2000,
            )
            print(f"✅ Работает: {model}")
            return response

        except Exception as e:
            error_str = str(e)
            print(f"❌ {model} не работает: {error_str[:100]}")

            # Если неверный ключ — нет смысла пробовать дальше
            if "401" in error_str and "User not found" in error_str:
                raise Exception("❌ Неверный API ключ OpenRouter. Проверьте OPENROUTER_API_KEY")

            last_error = e
            continue

    raise Exception(f"Все модели недоступны: {last_error}")


async def stream_analyze(query: str, history: list, sites_data: list = None):
    """Стриминг ответа"""
    messages = build_messages(query, history, sites_data)

    try:
        stream = await try_models(messages, stream=True)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
    except Exception as e:
        yield f"\n\n❌ Ошибка: {str(e)}"


async def decide_and_analyze(query: str, history: list, sites_data: list = None) -> dict:
    """Обычный запрос без стриминга"""
    messages = build_messages(query, history, sites_data)

    try:
        response = await try_models(messages, stream=False)
        answer = response.choices[0].message.content or ""

        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = response.usage.total_tokens or 0

        return {"answer": answer, "tokens_used": tokens_used}

    except Exception as e:
        print(f"❌ Все модели недоступны: {e}")
        return {
            "answer": "Все AI модели сейчас перегружены. Попробуйте через минуту.",
            "tokens_used": 0
        }