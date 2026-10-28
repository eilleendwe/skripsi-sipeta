"""Guard halaman: wajib login dan cek peran"""

import streamlit as st

from services import auth, roles
from ui import layout, nav

_NEXT_KEY = "_next_page"


def require_login() -> dict:
    """Mengembalikan pengguna yang login, atau pindah ke halaman Masuk kalau belum login"""
    user = auth.current_user()
    if user is None:
        # Ingat halaman tujuan supaya setelah login langsung dibawa ke sana.
        st.session_state[_NEXT_KEY] = nav.current_key()
        st.switch_page(nav.path("login"))
    return user


def require_role(minimum: str) -> dict:
    """Seperti `require_login`, lalu hentikan halaman kalau perannya tidak cukup"""
    user = require_login()
    if not roles.has_role(user["role"], minimum):
        _access_denied(minimum)
        st.stop()
    return user


def redirect_if_logged_in() -> None:
    """Untuk halaman Masuk dan Daftar: pengguna yang sudah login langsung ke Beranda"""
    if auth.current_user() is not None:
        st.switch_page(nav.path("beranda"))


def pop_next_page() -> str:
    """Halaman tujuan setelah login (bawaannya Beranda)"""
    key = st.session_state.pop(_NEXT_KEY, None)
    if key is None or nav.get(key).get("min_role") is None:
        return "beranda"
    return key


def _access_denied(minimum: str) -> None:
    with layout.card("akses_ditolak"):
        st.html(
            '<div class="sp-denied">'
            '<span class="sp-denied-ic material-symbols-rounded">lock</span>'
            f"<h2>Akses ditolak — halaman ini khusus {roles.label(minimum)}</h2>"
            "<p>Akun Anda tidak memiliki hak untuk membuka halaman ini. "
            "Hubungi Super Admin jika Anda memerlukan akses.</p>"
            "</div>"
        )
        with st.container(horizontal=True, horizontal_alignment="center"):
            if st.button("Kembali ke Beranda", icon=":material/home:", type="primary", key="btn_denied_home"):
                st.switch_page(nav.path("beranda"))
