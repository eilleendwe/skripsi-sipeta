"""Kerangka halaman: CSS, judul, kartu, notifikasi, dan blok isi kartu"""

import base64
from contextlib import contextmanager
from html import escape
from pathlib import Path

import streamlit as st

from ui import tabs

_CSS_PATH = Path(__file__).with_name("styles.css")
# st.html membuang tag <svg>, jadi logo dipasang sebagai <img> dengan data URI.
_LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(Path(__file__).with_name("logo.svg").read_bytes()).decode()
LOGO_PNG = str(Path(__file__).with_name("logo.png"))  # favicon
_FLASH_KEY = "_flash"


def inject_css() -> None:
    st.html(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>")


def logo(kelas: str = "sp-logo") -> str:
    """Logo SIPETA dengan kelas CSS `kelas` (ukurannya diatur di styles.css)"""
    return f'<img class="{kelas}" src="{_LOGO_URI}" alt="Logo SIPETA">'


def page_header(title: str, caption: str | None = None) -> None:
    st.title(title, anchor=False)
    if caption:
        st.caption(caption)


@contextmanager
def card(key: str, group: str | None = None, role: str | None = None):
    """
    Kartu putih bersudut bulat. Kalau `group` diisi, bagian atas kartu berisi tab grup itu
    (sesuai `role`).
    """
    with st.container(border=True, key=f"card_{key}"):
        if group:
            tabs.tab_bar(group, role)
        yield


def set_flash(message: str, icon: str | None = None) -> None:
    """Notifikasi yang muncul di rerun berikutnya (tetap muncul walau pindah halaman)"""
    st.session_state[_FLASH_KEY] = (message, icon)


def show_flash() -> None:
    flash = st.session_state.pop(_FLASH_KEY, None)
    if flash:
        st.toast(flash[0], icon=flash[1])


# --- Blok isi kartu -----------------------------------------------------------

def section(title: str, container=st) -> None:
    """Judul bagian kecil di dalam kartu"""
    container.html(f'<div class="sp-section">{escape(title)}</div>')


def empty_state(message: str, icon: str = "info") -> None:
    st.html(
        '<div class="sp-empty">'
        f'<span class="sp-empty-ic material-symbols-rounded">{escape(icon)}</span>'
        f"<div>{escape(message)}</div></div>"
    )


def metric_cards(items: list[tuple[str, str]], container=st, min_width: int = 110) -> None:
    """
    Deretan kartu metrik (label, nilai). Kartu turun baris kalau lebarnya di bawah `min_width` px.
    """
    kartu = "".join(
        f'<div class="sp-metric"><div class="sp-metric-label">{escape(label)}</div>'
        f'<div class="sp-metric-value">{escape(nilai)}</div></div>'
        for label, nilai in items
    )
    container.html(
        f'<div class="sp-metrics" style="grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))">'
        f"{kartu}</div>"
    )


def facts(rows: list[tuple[str, str]], container=st) -> None:
    """Daftar 'Label: nilai'"""
    isi = "".join(f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>" for k, v in rows)
    container.html(f'<dl class="sp-facts">{isi}</dl>')


def legend(items: list[tuple[str, tuple[int, int, int]]], container=st) -> None:
    """Legenda peta: (label, warna RGB)"""
    isi = "".join(
        f'<span class="sp-legend-item"><span class="sp-swatch" style="background:rgb({r},{g},{b})"></span>'
        f"{escape(label)}</span>"
        for label, (r, g, b) in items
    )
    container.html(f'<div class="sp-legend">{isi}</div>')
