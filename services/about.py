"""Data halaman Tentang, dibaca dari data/authors.json"""

import json
from pathlib import Path

from services.errors import ServiceError

_BERKAS = Path(__file__).resolve().parent.parent / "data" / "authors.json"


def get_about() -> dict:
    """
    Isi authors.json. Kolom yang tidak diisi tetap ada, isinya teks kosong.
    """
    try:
        data = json.loads(_BERKAS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ServiceError("Data halaman Tentang tidak dapat dibaca. Periksa berkas data/authors.json.")

    def isi(d: dict, *kunci: str) -> dict:
        return {k: str(d.get(k) or "").strip() for k in kunci}

    return {
        **isi(data, "judul", "institusi"),
        "penulis": isi(data.get("penulis") or {}, "nama", "nim", "prodi", "email"),
        "pembimbing": [isi(p, "nama", "peran") for p in data.get("pembimbing", [])],
        "sumber_data": [isi(s, "nama", "keterangan", "url") for s in data.get("sumber_data", [])],
    }
