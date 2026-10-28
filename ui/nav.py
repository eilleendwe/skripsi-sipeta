"""Daftar semua halaman, dipakai app.py, sidebar, tab, dan guard"""

import streamlit as st

from services import roles

_STATE_KEY = "_page_key"

# min_role kosong = halaman publik. 
# group = halaman yang tampil sebagai tab di satu kartu.
# sidebar False = halaman penuh tanpa sidebar.
PAGES = [
    {"key": "landing", "file": "pages/landing.py", "title": "SIPETA", "url": "", "sidebar": False},
    {"key": "login", "file": "pages/login.py", "title": "Masuk", "url": "masuk", "sidebar": False},
    {"key": "register", "file": "pages/register.py", "title": "Daftar", "url": "daftar", "sidebar": False},
    {"key": "beranda", "file": "pages/beranda.py", "title": "Beranda", "url": "beranda",
     "min_role": roles.USER, "icon": ":material/home:", "menu": "Beranda"},
    {"key": "klasifikasi_latih", "file": "pages/klasifikasi_latih.py", "title": "Klasifikasi",
     "url": "klasifikasi-latih", "min_role": roles.ADMIN, "group": "klasifikasi",
     "menu": "Latih model", "tab": "Latih Model"},
    {"key": "klasifikasi_citra", "file": "pages/klasifikasi_citra.py", "title": "Klasifikasi",
     "url": "klasifikasi-citra", "min_role": roles.USER, "group": "klasifikasi",
     "menu": "Klasifikasi citra", "tab": "Klasifikasi Citra"},
    {"key": "prediksi_latih", "file": "pages/prediksi_latih.py", "title": "Prediksi",
     "url": "prediksi-latih", "min_role": roles.ADMIN, "group": "prediksi",
     "menu": "Latih model", "tab": "Latih Model"},
    {"key": "prediksi_citra", "file": "pages/prediksi_citra.py", "title": "Prediksi",
     "url": "prediksi-citra", "min_role": roles.USER, "group": "prediksi",
     "menu": "Prediksi citra", "tab": "Prediksi Citra"},
    {"key": "pemantauan", "file": "pages/pemantauan.py", "title": "Pemantauan", "url": "pemantauan",
     "min_role": roles.USER, "icon": ":material/monitoring:", "menu": "Pemantauan"},
    {"key": "panduan", "file": "pages/panduan.py", "title": "Panduan Pengguna", "url": "panduan",
     "min_role": roles.USER, "icon": ":material/menu_book:", "menu": "Panduan Pengguna"},
    {"key": "tentang", "file": "pages/tentang.py", "title": "Tentang", "url": "tentang",
     "min_role": roles.USER, "icon": ":material/info:", "menu": "Tentang"},
    {"key": "kelola_pengguna", "file": "pages/kelola_pengguna.py", "title": "Kelola Pengguna",
     "url": "kelola-pengguna", "min_role": roles.SUPERADMIN, "icon": ":material/manage_accounts:",
     "menu": "Kelola Pengguna"},
]

# Menu induk untuk halaman bergrup dengan default adalaha halaman prediksi dan klasifikasi citra.
GROUPS = {
    "klasifikasi": {"label": "Klasifikasi", "icon": ":material/layers:", "default": "klasifikasi_citra"},
    "prediksi": {"label": "Prediksi", "icon": ":material/timeline:", "default": "prediksi_citra"},
}

# Urutan menu di sidebar
MENU = ["beranda", "klasifikasi", "prediksi", "pemantauan", "panduan", "tentang", "kelola_pengguna"]


def get(key: str) -> dict:
    for page in PAGES:
        if page["key"] == key:
            return page
    raise KeyError(key)


def path(key: str) -> str:
    """Path berkas halaman, untuk `st.page_link` dan `st.switch_page`"""
    return get(key)["file"]


def allowed(key: str, role: str | None) -> bool:
    min_role = get(key).get("min_role")
    return min_role is None or roles.has_role(role, min_role)


def group_pages(group: str, role: str | None) -> list[dict]:
    """Halaman di satu grup yang boleh dibuka `role`, urut sesuai PAGES"""
    return [p for p in PAGES if p.get("group") == group and allowed(p["key"], role)]


def run_navigation():
    """
    Daftarkan semua halaman dan kembalikan (key halaman aktif, halaman).
    Halaman yang tidak boleh dibuka peran saat ini tetap didaftarkan, supaya alamatnya
    menampilkan "Akses ditolak" dari guard, bukan "Page not found". Menu sidebar disaring terpisah.
    """
    pages = {}
    for p in PAGES:
        judul = "SIPETA" if p["key"] == "landing" else f"{p['title']} · SIPETA"
        pages[p["key"]] = st.Page(p["file"], title=judul, url_path=p["url"], default=(p["key"] == "landing"))
    current = st.navigation(list(pages.values()), position="hidden")
    key = next((k for k, page in pages.items() if page is current), None)
    if key is None:
        key = next((k for k, page in pages.items() if page.url_path == current.url_path), "landing")
    st.session_state[_STATE_KEY] = key
    return key, current


def current_key() -> str:
    return st.session_state.get(_STATE_KEY, "landing")
