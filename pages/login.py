import streamlit as st

from services import auth
from services.errors import ServiceError
from ui import guard, layout, nav

guard.redirect_if_logged_in()

_, tengah, _ = st.columns([1, 1.15, 1])
with tengah:
    st.page_link(nav.path("landing"), label="Kembali", icon=":material/arrow_back:")
    with layout.card("auth_login"):
        st.html(
            f'<div class="sp-auth-brand">{layout.logo()}SIPETA</div>'
            '<div class="sp-auth-title">Masuk</div>'
        )
        with st.form("form_login", border=False):
            email = st.text_input("Email", key="login_email", autocomplete="username")
            password = st.text_input("Kata sandi", type="password", key="login_password",
                                     autocomplete="current-password")
            submitted = st.form_submit_button("Masuk", type="primary", width="stretch")

        if submitted:
            try:
                auth.login(email, password)
            except ServiceError as e:
                st.error(e.message, icon=":material/error:")
            else:
                st.switch_page(nav.path(guard.pop_next_page()))

        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.markdown("Belum punya akun?")
            st.page_link(nav.path("register"), label="Daftar")
