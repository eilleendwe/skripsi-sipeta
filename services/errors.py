from core.jobs import ProsesError

# Error yang pesannya boleh langsung ditampilkan ke pengguna. Sama dengan ProsesError di core,
# jadi error dari core bisa langsung ditangkap halaman dengan `except ServiceError`.
ServiceError = ProsesError
