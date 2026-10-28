class ServiceError(Exception):
    """Error yang pesannya boleh langsung ditampilkan ke pengguna"""

    def __init__(self, message: str, fields: dict[str, str] | None = None):
        super().__init__(message)
        self.message = message
        # Pesan per field form (kunci = nama field), kalau ada.
        self.fields = fields or {}
