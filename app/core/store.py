"""
In-memory job store.
В production замени на Redis:
    import redis.asyncio as redis
    r = redis.from_url("redis://localhost")
"""
from typing import Any

job_store: dict[str, Any] = {}
