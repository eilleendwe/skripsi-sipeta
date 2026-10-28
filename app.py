"""Titik masuk SIPETA"""

import streamlit as st

from core import operasi
from services import auth
from services.errors import ServiceError
from ui import layout, nav, sidebar


@st.cache_resource(show_spinner=False)
def _siapkan() -> None:
    """Jalan sekali per proses: cek pengaturan, bersihkan berkas sementara, buat folder storage"""
    operasi.siapkan()


st.set_page_config(
    page_icon=layout.LOGO_PNG,
    layout="wide",
    initial_sidebar_state="expanded",
)
layout.inject_css()

try:
    _siapkan()
except RuntimeError as e:  # pengaturan database atau storage tidak valid
    st.error(str(e), icon=":material/error:")
    st.stop()

page_key, page = nav.run_navigation()

# Sidebar cuma tampil di halaman aplikasi dan kalau sudah login. Hak akses tiap halaman
# dicek guard di dalam halaman itu sendiri.
try:
    user = auth.current_user()
except ServiceError as e:
    st.error(e.message, icon=":material/error:")
    st.stop()
if user and nav.get(page_key).get("sidebar", True):
    sidebar.render(page_key, user)

layout.show_flash()
page.run()
