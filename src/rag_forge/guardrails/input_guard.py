class InputGuard:
    def __init__(self, max_length: int = 500):
        self._max_length = max_length

    def validate(self, query: str) -> tuple[bool, str]:
        """Return is_valid, error_message. Empty strings means valid"""
        if not query or not query.strip():
            return False, "Query cannot be empty"
        if len(query) > self._max_length:
            return False, f"Query too long ({len(query)} chars, max {self._max_length})"
        return True, ""