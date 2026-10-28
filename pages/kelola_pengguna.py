from html import escape

import streamlit as st

from services import auth, roles
from services.errors import ServiceError
from ui import badges, guard, layout

aktor = guard.require_role(roles.SUPERADMIN)

_PENDING = "_role_pending" 
_NONCE = "_role_nonce"
_KOLOM = [2.2, 2.6, 1.3, 1.3, 1.7]


def _select_key(user_id: str) -> str:
    return f"role_select_{user_id}_{st.session_state.get(_NONCE, 0)}"


def _on_pick(user_id: str) -> None:
    st.session_state[_PENDING] = (user_id, st.session_state[_select_key(user_id)])


def _selesai() -> None:
    """Tutup konfirmasi dan kembalikan dropdown ke peran yang tersimpan"""
    st.session_state.pop(_PENDING, None)
    st.session_state[_NONCE] = st.session_state.get(_NONCE, 0) + 1


@st.dialog("Ubah peran", on_dismiss=_selesai)
def _konfirmasi(target: dict, peran_baru: str) -> None:
    st.markdown(
        f"Ubah peran **{target['nama']}** dari **{roles.label(target['role'])}** "
        f"ke **{roles.label(peran_baru)}**?"
    )
    st.caption("Perubahan berlaku langsung, termasuk jika pengguna tersebut sedang masuk.")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("Batal", key="role_batal"):
            _selesai()
            st.rerun()
        if st.button("Ubah peran", type="primary", key="role_ok"):
            try:
                auth.change_role(target["id"], peran_baru)
            except ServiceError as e:
                layout.set_flash(e.message, ":material/error:")
            else:
                layout.set_flash(
                    f"Peran {target['nama']} diubah menjadi {roles.label(peran_baru)}.",
                    ":material/check_circle:",
                )
            _selesai()
            st.rerun()


layout.page_header("Kelola Pengguna")
with layout.card("kelola_pengguna"):
    c_cari, c_peran, _ = st.columns([2.4, 1.4, 2.2])
    cari = c_cari.text_input("Cari", placeholder="Nama atau email", icon=":material/search:", key="user_cari")
    filter_peran = c_peran.selectbox(
        "Peran", [None, *roles.ORDER], key="user_filter",
        format_func=lambda r: "Semua peran" if r is None else roles.label(r),
    )

    pengguna = auth.list_users(cari, filter_peran)
    st.caption(f"{len(pengguna)} pengguna")

    with st.container(key="user_head"):
        kolom = st.columns(_KOLOM, vertical_alignment="center")
        for c, judul in zip(kolom, ["Nama", "Email", "Peran", "Tanggal daftar", "Ubah peran"]):
            c.html(f'<div class="sp-section">{judul}</div>')

    if not pengguna:
        st.info("Tidak ada pengguna yang cocok dengan pencarian.", icon=":material/person_search:")

    for u in pengguna:
        halangan = auth.role_change_blocker(aktor, u)
        with st.container(key=f"user_row_{u['id']}"):
            c_nama, c_email, c_role, c_tgl, c_ubah = st.columns(_KOLOM, vertical_alignment="center")
            anda = " (Anda)" if u["id"] == aktor["id"] else ""
            c_nama.html(f'<div class="sp-cell">{escape(u["nama"])}{anda}</div>')
            c_email.html(f'<div class="sp-cell sp-cell-soft">{escape(u["email"])}</div>')
            c_role.html(f'<div class="sp-cell">{badges.role_badge_html(u["role"])}</div>')
            c_tgl.html(f'<div class="sp-cell sp-cell-soft">{u["tanggal_daftar"]:%d-%m-%Y}</div>')
            c_ubah.selectbox(
                f"Peran {u['nama']}", roles.ORDER,
                index=roles.ORDER.index(u["role"]),
                format_func=roles.label,
                key=_select_key(u["id"]),
                label_visibility="collapsed",
                disabled=halangan is not None,
                help=halangan,
                on_change=_on_pick,
                args=(u["id"],),
            )

pending = st.session_state.get(_PENDING)
if pending:
    target = next((u for u in auth.list_users() if u["id"] == pending[0]), None)
    if target is None or target["role"] == pending[1]:
        _selesai()
    else:
        _konfirmasi(target, pending[1])
