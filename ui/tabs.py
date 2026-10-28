"""
Tab di bagian atas kartu untuk halaman Klasifikasi dan Prediksi.
"""

import streamlit as st

from ui import nav


def tab_bar(group: str, role: str | None) -> None:
    aktif = nav.current_key()
    with st.container(key=f"tabbar_{group}", horizontal=True, gap="medium"):
        for page in nav.group_pages(group, role):
            state = "on" if page["key"] == aktif else "off"
            with st.container(key=f"tab_{state}_{page['key']}", width="content"):
                st.page_link(page["file"], label=page["tab"])
