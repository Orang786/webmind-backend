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

def safe_parse_json(text: str) -> dict:
    """Надёжный парсинг JSON даже если Gemini вернул кривой текст"""
    
    # Убираем markdown блоки если есть
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Попытка 1: прямой парсинг
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Попытка 2: чистим плохие escape-символы
    try:
        # Заменяем невалидные escape последовательности
        cleaned = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Попытка 3: ищем JSON внутри текста через регулярку
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass

    # Попытка 4: если всё сломалось - возвращаем текст как answer
    return {"answer": text}


async def decide_and_analyze(query: str, history: list, sites_data: list = None) -> dict:
    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

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
4. Отвечай развёрнуто и структурированно.

ВАЖНО: Отвечай строго в JSON формате:
{"answer": "текст ответа в markdown"}

Никаких дополнительных полей. Только поле answer."""

    user_content = f"{context_str}\n\nВопрос: {query}" if context_str else query
    messages.append({"role": "user", "content": user_content})

    try:
        response = await client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            messages=[
                {"role": "system", "content": system_prompt},
                *messages
            ],
            response_format={"type": "json_object"}
        )

        raw = response.choices[0].message.content
        print(f"📥 Ответ AI (первые 200 символов): {raw[:200]}")

        # Используем надёжный парсинг
        data = safe_parse_json(raw)

        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = response.usage.total_tokens or 0

        return {
            "answer": data.get("answer", raw),
            "tokens_used": tokens_used
        }

    except Exception as e:
        print(f"❌ Ошибка AI: {e}")
        return {
            "answer": "Произошла ошибка при обработке запроса. Попробуй ещё раз.",
            "tokens_used": 0
        }