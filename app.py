import shutil

import streamlit as st

from core import storage
from core.config import CITRA_DIR, HASIL_DIR, MODEL_DIR, TMP_DIR, periksa_db_url
from services import auth
from services.errors import ServiceError
from ui import layout, nav, sidebar


@st.cache_resource(show_spinner=False)
def _siapkan() -> None:
    """Sekali saat app start: cek pengaturan, buang sisa berkas sementara, buat folder"""
    periksa_db_url()
    storage.periksa()
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    for folder in (TMP_DIR, MODEL_DIR, HASIL_DIR, CITRA_DIR):
        folder.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_icon=layout.LOGO_PNG,
    layout="wide",
    initial_sidebar_state="expanded",
)
layout.inject_css()

try:
    _siapkan()
except RuntimeError as e:  
    st.error(str(e), icon=":material/error:")
    st.stop()

page_key, page = nav.run_navigation()

# Sidebar cuma tampil di halaman aplikasi dan kalau sudah login. Hak akses tiap halaman dicek guard di dalam halaman itu sendiri.
try:
    user = auth.current_user()
except ServiceError as e:
    st.error(e.message, icon=":material/error:")
    st.stop()
if user and nav.get(page_key).get("sidebar", True):
    sidebar.render(page_key, user)

layout.show_flash()
page.run()
