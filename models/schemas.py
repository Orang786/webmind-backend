from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class SearchRequest(BaseModel):
    query: str
    history: List[Message] = []

class SiteInfo(BaseModel):
    url: str
    title: str
    snippet: str

class SearchResponse(BaseModel):
    answer: str
    sources: List[SiteInfo] = []
    is_search_performed: bool = False
    tokens_used: int = 0