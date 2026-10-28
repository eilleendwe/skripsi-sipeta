"""Akses PostgreSQL lewat connection pool"""

from functools import lru_cache

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from core.config import DB_URL


@lru_cache(maxsize=1)
def _pool() -> ConnectionPool:
    return ConnectionPool(DB_URL, min_size=1, max_size=4, timeout=15, open=True,
                          kwargs={"row_factory": dict_row, "prepare_threshold": None},
                          check=ConnectionPool.check_connection)


def fetch_all(sql: str, params=None) -> list[dict]:
    with _pool().connection() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params=None) -> dict | None:
    with _pool().connection() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params=None) -> None:
    with _pool().connection() as conn:
        conn.execute(sql, params)


def json(value) -> Jsonb:
    """Bungkus dict/list supaya tersimpan sebagai json"""
    return Jsonb(value)
