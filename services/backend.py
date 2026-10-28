"""
Jembatan services/ ke core/: menjalankan fungsi core dan mengubah error-nya jadi ServiceError.

Hasilnya dibentuk ulang seperti JSON (tanggal jadi teks ISO, tuple jadi list, kunci dict jadi teks),
supaya bentuk data yang diterima services selalu sama.
"""

import json
import logging
import time
from datetime import date, datetime

import numpy as np
import psycopg
from psycopg_pool import PoolTimeout

from core import operasi
from core.config import DB_URL, tujuan_db
from core.jobs import ProsesError
from services.errors import ServiceError

log = logging.getLogger("sipeta")


def _json(nilai):
    def lain(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(f"{type(o).__name__} tidak bisa diubah ke JSON")
    return json.loads(json.dumps(nilai, default=lain))


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


def panggil(fungsi, *args, **kwargs):
    """
    Jalankan `fungsi(*args, **kwargs)`. Error-nya diubah jadi ServiceError yang aman ditampilkan.
    """
    try:
        hasil = fungsi(*args, **kwargs)
    except ProsesError as e:
        raise ServiceError(str(e), e.fields) from None
    except PoolTimeout:
        raise ServiceError(f"Database tidak dapat diakses ({tujuan_db(DB_URL)}): {_sebab_koneksi_gagal()}") from None
    except psycopg.Error as e:
        pesan = str(e).strip().splitlines()[0] if str(e).strip() else e.__class__.__name__
        raise ServiceError(f"Database tidak dapat diakses: {pesan}") from None
    except FileNotFoundError as e:
        raise ServiceError(f"Berkas tidak ditemukan di storage: {e}") from None
    except Exception as e:  # error tak terduga: catat lengkap di log, tampilkan ringkas
        log.exception("Gagal menjalankan %s", getattr(fungsi, "__name__", fungsi))
        raise ServiceError(f"Terjadi error saat memproses: {e}") from None
    return hasil if hasil is None or isinstance(hasil, bytes) else _json(hasil)


def tunggu_job(job_id: str, bar, jeda: float = 1.0):
    """
    Cek status job tiap `jeda` detik sambil update progress bar. Kalau selesai, hasil job-nya dikembalikan.
    """
    while True:
        job = panggil(operasi.status_job, job_id)
        bar.progress(job["progress"], text=job["label"])
        if job["status"] == "selesai":
            return job["hasil"]
        if job["status"] == "gagal":
            raise ServiceError(job["pesan"])
        time.sleep(jeda)
