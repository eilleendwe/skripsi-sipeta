import streamlit as st

from services import auth
from services.errors import ServiceError
from ui import guard, layout, nav

guard.redirect_if_logged_in()

_, tengah, _ = st.columns([1, 1.15, 1])
with tengah:
    st.page_link(nav.path("landing"), label="Kembali", icon=":material/arrow_back:")
    with layout.card("auth_register"):
        st.html(
            f'<div class="sp-auth-brand">{layout.logo()}SIPETA</div>'
            '<div class="sp-auth-title">Daftar</div>'
        )
        # Tempat pesan error di bawah tiap field, diisi setelah validasi.
        slots = {}
        with st.form("form_register", border=False):
            nama = st.text_input("Nama lengkap", key="reg_nama", autocomplete="name")
            slots["nama"] = st.empty()
            email = st.text_input("Email", key="reg_email", autocomplete="email")
            slots["email"] = st.empty()
            # Aturan kata sandi ditulis di bawah label supaya langsung kelihatan.
            password = st.text_input(f"Kata sandi  \n:small[:gray[{auth.ATURAN_SANDI}]]", type="password",
                                     key="reg_password", autocomplete="new-password")
            slots["password"] = st.empty()
            konfirmasi = st.text_input("Konfirmasi kata sandi", type="password", key="reg_konfirmasi",
                                       autocomplete="new-password")
            slots["konfirmasi"] = st.empty()
            submitted = st.form_submit_button("Daftar", type="primary", width="stretch")

        if submitted:
            try:
                auth.register(nama, email, password, konfirmasi)
            except ServiceError as e:
                for field, pesan in e.fields.items():
                    slots[field].html(f'<div class="sp-field-error">{pesan}</div>')
                if not e.fields:
                    st.error(e.message, icon=":material/error:")
            else:
                layout.set_flash("Akun dibuat. Selamat datang di SIPETA.", ":material/check_circle:")
                st.switch_page(nav.path("beranda"))

        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            st.markdown("Sudah punya akun?")
            st.page_link(nav.path("login"), label="Masuk")
