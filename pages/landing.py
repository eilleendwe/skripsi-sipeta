from datetime import date

import streamlit as st

from services import auth
from services.errors import ServiceError
from ui import layout, nav

try:
    user = auth.current_user()
except ServiceError:
    user = None

# Lebar khusus halaman ini (hilang lagi saat pindah halaman). Latar grid ada di styles.css.
st.html("""<style>
/* Footer tetap di bawah layar walau isi halaman pendek. */
[data-testid="stMainBlockContainer"] {
  max-width: 1180px;
  padding: 1.1rem 2rem 1rem;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}
[data-testid="stElementContainer"]:has(.sp-land-footer) { margin-top: auto; }
</style>""")

with st.container(key="landing_nav", horizontal=True, horizontal_alignment="distribute",
                  vertical_alignment="center"):
    st.html(f'<div class="sp-land-brand">{layout.logo()}SIPETA</div>')
    with st.container(horizontal=True, gap="small", vertical_alignment="center", width="content"):
        if user:
            st.page_link(nav.path("beranda"), label="Buka Beranda", icon=":material/home:")
        else:
            st.page_link(nav.path("login"), label="Masuk")
            if st.button("Daftar", key="nav_daftar", type="secondary"):
                st.switch_page(nav.path("register"))

with st.container(key="landing_hero"):
    c_teks, c_logo = st.columns([1.05, 1], gap="large", vertical_alignment="center")
    with c_teks:
        st.html(
            '<div class="sp-hero-kicker">Sentinel-2 Level-2A · Random Forest · ConvLSTM</div>'
            '<h1 class="sp-hero-title">SIPETA: Sistem Pemantauan Tutupan Lahan Terbangun</h1>'
        )
        with st.container(horizontal=True, gap="small"):
            if user:
                if st.button("Buka Beranda", type="primary", key="btn_mulai", icon=":material/arrow_forward:"):
                    st.switch_page(nav.path("beranda"))
            else:
                if st.button("Masuk", type="primary", key="btn_mulai", icon=":material/login:"):
                    st.switch_page(nav.path("login"))
        st.html('<div class="sp-hero-wilayah">Kota Tangerang · Kota Tangerang Selatan · Kabupaten Tangerang</div>')

    c_logo.html(f'<div class="sp-hero-logo">{layout.logo("sp-hero-logo-img")}</div>')

st.html(
    '<footer class="sp-land-footer">'
    f"<span>© {date.today().year} · SIPETA</span>"
    "</footer>"
)
