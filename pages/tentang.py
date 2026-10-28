from html import escape

import streamlit as st

from services import about, roles
from services.errors import ServiceError
from ui import guard, layout

guard.require_role(roles.USER)


def _rincian(rows: list[tuple[str, str]]) -> str:
    """Daftar 'Label: nilai', baris yang nilainya kosong dilewati"""
    isi = "".join(f"<div><dt>{escape(k)}</dt><dd>{escape(v)}</dd></div>" for k, v in rows if v)
    return f'<dl class="sp-facts">{isi}</dl>'


def _penulis(p: dict) -> str:
    return (
        '<div class="sp-orang"><div class="sp-orang-peran">Penulis</div>'
        f'<div class="sp-orang-nama">{escape(p["nama"])}</div>'
        + _rincian([("NIM", p["nim"]), ("Program studi", p["prodi"]), ("Email", p["email"])])
        + "</div>"
    )


def _pembimbing(daftar: list[dict]) -> str:
    isi = "".join(
        f'<div class="sp-pembimbing"><div class="sp-orang-peran">{escape(d["peran"])}</div>'
        f'<div class="sp-orang-nama">{escape(d["nama"])}</div></div>'
        for d in daftar
    )
    return f'<div class="sp-orang">{isi}</div>'


layout.page_header("Tentang")
with layout.card("tentang"):
    try:
        data = about.get_about()
    except ServiceError as e:
        st.error(e.message, icon=":material/error:")
        st.stop()

    layout.section("Tugas akhir")
    prodi = data["penulis"]["prodi"]
    asal = ", ".join(filter(None, [prodi and f"Program Studi {prodi}", data["institusi"]]))
    st.html('<p class="sp-body">SIPETA dibangun sebagai tugas akhir (skripsi)'
            f'{" di " + escape(asal) if asal else ""} dengan judul:</p>')
    if data["judul"]:
        st.html(f'<div class="sp-judul">{escape(data["judul"])}</div>')

    st.divider()
    layout.section("Penulis dan dosen pembimbing")
    if not data["penulis"]["nama"] and not data["pembimbing"]:
        layout.empty_state("Data penulis belum diisi.", "groups")
    else:
        st.html('<div class="sp-orang-grid">' + (_penulis(data["penulis"]) if data["penulis"]["nama"] else "")
                + (_pembimbing(data["pembimbing"]) if data["pembimbing"] else "") + "</div>")

    if data["sumber_data"]:
        st.divider()
        layout.section("Sumber data")
        for i, s in enumerate(data["sumber_data"]):
            with st.container(key=f"tentang_sumber_{i}"):
                c_teks, c_link = st.columns([3, 1], vertical_alignment="center", gap="medium")
                c_teks.html(f'<div class="sp-sumber"><b>{escape(s["nama"])}</b>'
                            f'<span>{escape(s["keterangan"])}</span></div>')
                if s["url"]:
                    c_link.page_link(s["url"], label="Buka situs", icon=":material/open_in_new:")
