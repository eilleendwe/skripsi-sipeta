"""Akses PostgreSQL lewat connection pool"""

from contextlib import contextmanager
from functools import lru_cache

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from core.config import DB_URL, tujuan_db
from core.jobs import ProsesError


@lru_cache(maxsize=1)
def _pool() -> ConnectionPool:
    return ConnectionPool(DB_URL, min_size=1, max_size=4, timeout=15, open=True,
                          kwargs={"row_factory": dict_row, "prepare_threshold": None},
                          check=ConnectionPool.check_connection)


def _sebab_koneksi_gagal() -> str:
    """
    Pool cuma melapor "couldn't get a connection after ... sec". Coba satu koneksi langsung untuk tahu
    sebab aslinya (host salah, password salah, project Supabase di-pause, dll.).
    """
    try:
        psycopg.connect(DB_URL, connect_timeout=8).close()
    except psycopg.Error as e:
        baris = [b.strip() for b in str(e).strip().splitlines() if b.strip()]
        return baris[-1] if baris else e.__class__.__name__
    return "koneksi sempat gagal, coba lagi"


@contextmanager
def _koneksi():
    """Koneksi dari pool. Error database diubah jadi pesan yang aman ditampilkan."""
    try:
        with _pool().connection() as conn:
            yield conn
    except PoolTimeout:
        raise ProsesError(f"Database tidak dapat diakses ({tujuan_db(DB_URL)}): {_sebab_koneksi_gagal()}") from None
    except psycopg.Error as e:
        pesan = str(e).strip().splitlines()[0] if str(e).strip() else e.__class__.__name__
        raise ProsesError(f"Database tidak dapat diakses: {pesan}") from None


def fetch_all(sql: str, params=None) -> list[dict]:
    with _koneksi() as conn:
        return conn.execute(sql, params).fetchall()


def fetch_one(sql: str, params=None) -> dict | None:
    with _koneksi() as conn:
        return conn.execute(sql, params).fetchone()


def execute(sql: str, params=None) -> None:
    with _koneksi() as conn:
        conn.execute(sql, params)


def json(value) -> Jsonb:
    """Bungkus dict/list supaya tersimpan sebagai json"""
    return Jsonb(value)
