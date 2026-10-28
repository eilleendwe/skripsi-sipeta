"""
Job latar belakang untuk proses yang lama (latih model, klasifikasi, prediksi).
Job jalan di thread dalam proses, jadi tetap lanjut walau pengguna pindah halaman. Status job disimpan di memori, jadi hilang kalau app di-restart.
"""

import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("sipeta")

_latih = ThreadPoolExecutor(max_workers=1)
_antrean = {"latih": _latih, "latih_prediksi": _latih,
            "klasifikasi": ThreadPoolExecutor(max_workers=1), "prediksi": ThreadPoolExecutor(max_workers=1)}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


class ProsesError(Exception):
    """Error yang pesannya boleh langsung ditampilkan ke pengguna. `fields` = pesan per field form."""

    def __init__(self, pesan: str, fields: dict[str, str] | None = None):
        super().__init__(pesan)
        self.fields = fields or {}


def buat(jenis: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "jenis": jenis, "status": "antre", "progress": 0.0,
                         "label": "Menunggu giliran…", "hasil": None, "pesan": None}
    return job_id


def mulai(job_id: str, fungsi, *args) -> None:
    """Masukkan `fungsi(job_id, *args)` ke antrean. Nilai kembaliannya jadi hasil job."""
    _antrean[get(job_id)["jenis"]].submit(_jalankan, job_id, fungsi, args)


def _jalankan(job_id, fungsi, args):
    update(job_id, 0.0, "Memulai…", status="berjalan")
    try:
        hasil = fungsi(job_id, *args)
    except ProsesError as e:
        _set(job_id, status="gagal", pesan=str(e))
    except Exception as e:  # error tak terduga: catat lengkap di log
        log.error("Job %s gagal:\n%s", job_id, traceback.format_exc())
        _set(job_id, status="gagal", pesan=f"Terjadi error saat memproses: {e}")
    else:
        _set(job_id, status="selesai", progress=1.0, label="Selesai", hasil=hasil)


def _set(job_id, **nilai):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(nilai)


def update(job_id: str, progress: float, label: str, status: str | None = None) -> None:
    nilai = {"progress": round(min(max(progress, 0.0), 1.0), 4), "label": label}
    if status:
        nilai["status"] = status
    _set(job_id, **nilai)


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def hapus(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
