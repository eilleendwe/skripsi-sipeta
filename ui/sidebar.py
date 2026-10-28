"""Sidebar: logo, menu sesuai peran, info pengguna, dan tombol keluar"""

from html import escape

import streamlit as st

from services import auth
from ui import badges, layout, nav


def _link(slot: str, target: str, label: str, icon: str | None, active: bool, sub: bool = False) -> None:
    kind = "sub" if sub else "top"
    state = "on" if active else "off"
    with st.container(key=f"nav_{kind}_{state}_{slot}"):
        st.page_link(nav.path(target), label=label, icon=icon, width="stretch")


def _menu(current_key: str, role: str) -> None:
    current = nav.get(current_key)
    for item in nav.MENU:
        if item in nav.GROUPS:
            children = nav.group_pages(item, role)
            if not children:
                continue
            group = nav.GROUPS[item]
            _link(item, group["default"], group["label"], group["icon"], active=current.get("group") == item)
            for child in children:
                on = child["key"] == current_key
                icon = ":material/radio_button_checked:" if on else ":material/radio_button_unchecked:"
                _link(child["key"], child["key"], child["menu"], icon, active=on, sub=True)
        elif nav.allowed(item, role):
            page = nav.get(item)
            _link(item, item, page["menu"], page["icon"], active=item == current_key)


def render(current_key: str, user: dict) -> None:
    with st.sidebar:
        st.html(
            '<div class="sp-brand">'
            f'<div class="sp-brand-name">{layout.logo()}SIPETA</div>'
            '<div class="sp-brand-sub">Sistem Pemantauan Tutupan Lahan Terbangun</div>'
            "</div>"
        )
        with st.container(key="sidenav", gap=None):
            _menu(current_key, user["role"])

        with st.container(key="sidefoot"):
            st.html(
                '<div class="sp-user">'
                f'<div class="sp-user-name">{escape(user["nama"])}</div>'
                f'<div class="sp-user-mail">{escape(user["email"])}</div>'
                f"{badges.role_badge_html(user['role'])}"
                "</div>"
            )
            if st.button("Keluar", icon=":material/logout:", key="btn_keluar", width="stretch"):
                auth.logout()
                layout.set_flash("Anda telah keluar.")
                st.switch_page(nav.path("landing"))
