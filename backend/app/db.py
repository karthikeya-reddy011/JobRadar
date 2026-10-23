from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import DATABASE_URL


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True)
