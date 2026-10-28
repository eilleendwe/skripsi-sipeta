"""Badge kecil (pill) untuk peran dan status"""

from html import escape

import streamlit as st

from services import roles

_ROLE_TONE = {
    roles.USER: "slate",
    roles.ADMIN: "teal",
    roles.SUPERADMIN: "amber",
}


def pill_html(text: str, tone: str = "slate", icon: str | None = None, wrap: bool = False) -> str:
    """
    HTML badge. `tone`: slate, teal, green, amber, red. `wrap` untuk teks panjang.
    """
    ikon = f'<span class="sp-pill-ic material-symbols-rounded">{escape(icon)}</span>' if icon else ""
    kelas = f"sp-pill sp-pill-{tone}" + (" sp-pill-wrap" if wrap else "")
    return f'<span class="{kelas}">{ikon}{escape(text)}</span>'


def pill(text: str, tone: str = "slate", icon: str | None = None) -> None:
    st.html(pill_html(text, tone, icon))


def role_badge_html(role: str) -> str:
    return pill_html(roles.label(role), _ROLE_TONE.get(role, "slate"))

