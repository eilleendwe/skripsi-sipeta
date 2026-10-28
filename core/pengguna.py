"""
Akun pengguna: daftar, masuk, dan ubah peran. Password disimpan sebagai hash PBKDF2-SHA256.
"""

import hashlib
import hmac
import secrets

from core import db
from core.jobs import ProsesError

PERAN = ("user", "admin", "superadmin")
_ITERASI = 200_000
_KOLOM = "SELECT id, nama, email, peran AS role, tanggal_daftar FROM pengguna "


def hash_sandi(sandi: str) -> str:
    garam = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", sandi.encode(), garam.encode(), _ITERASI).hex()
    return f"pbkdf2_sha256${_ITERASI}${garam}${h}"


def sandi_cocok(sandi: str, tersimpan: str) -> bool:
    _, iterasi, garam, h = tersimpan.split("$")
    baru = hashlib.pbkdf2_hmac("sha256", sandi.encode(), garam.encode(), int(iterasi)).hex()
    return hmac.compare_digest(baru, h)


def daftar(nama: str, email: str, password: str) -> dict:
    """
    Akun baru selalu berperan Pengguna. Isian form sudah dicek di services.auth.
    """
    if len(nama.strip()) > 100:
        raise ProsesError("Periksa kembali isian form.", {"nama": "Nama maksimal 100 karakter."})
    if len(email.strip()) > 150:
        raise ProsesError("Periksa kembali isian form.", {"email": "Email maksimal 150 karakter."})
    row = db.fetch_one(
        "INSERT INTO pengguna (nama, email, password_hash) VALUES (%s, %s, %s) "
        "ON CONFLICT (email) DO NOTHING RETURNING id",
        (nama.strip(), email.strip().lower(), hash_sandi(password)),
    )
    if row is None:
        raise ProsesError("Periksa kembali isian form.", {"email": "Email sudah terdaftar."})
    return detail(row["id"])


def masuk(email: str, password: str) -> dict:
    row = db.fetch_one("SELECT id, password_hash FROM pengguna WHERE email = %s", (email.strip().lower(),))
    if row is None or not sandi_cocok(password, row["password_hash"]):
        raise ProsesError("Email atau kata sandi salah.")
    return detail(row["id"])


def daftar_pengguna(cari: str = "", peran: str | None = None) -> list[dict]:
    pola = f"%{cari.strip().lower()}%"
    return db.fetch_all(
        _KOLOM + "WHERE (lower(nama) LIKE %(pola)s OR email LIKE %(pola)s) "
                 "AND (%(peran)s::text IS NULL OR peran = %(peran)s) ORDER BY nama",
        {"pola": pola, "peran": peran},
    )


def detail(pengguna_id: int) -> dict | None:
    """Data pengguna, atau None kalau akunnya sudah tidak ada"""
    return db.fetch_one(_KOLOM + "WHERE id = %s", (pengguna_id,))


def ubah_peran(pengguna_id: int, peran: str, oleh: int) -> dict:
    """
    `oleh` = id Super Admin yang mengubah. Aturannya dicek lagi di sini, tidak cuma di halaman.
    """
    if peran not in PERAN:
        raise ProsesError("Peran tidak dikenal.")
    aktor, target = detail(oleh), detail(pengguna_id)
    if aktor is None or aktor["role"] != "superadmin":
        raise ProsesError("Hanya Super Admin yang dapat mengubah peran.")
    if target is None:
        raise ProsesError("Pengguna tidak ditemukan.")
    if aktor["id"] == target["id"]:
        raise ProsesError("Anda tidak dapat mengubah peran sendiri.")
    if target["role"] == "superadmin" and peran != "superadmin":
        jumlah = db.fetch_one("SELECT count(*) AS n FROM pengguna WHERE peran = 'superadmin'")["n"]
        if jumlah <= 1:
            raise ProsesError("Super Admin terakhir tidak dapat diturunkan.")
    db.execute("UPDATE pengguna SET peran = %s WHERE id = %s", (peran, pengguna_id))
    return detail(pengguna_id)
