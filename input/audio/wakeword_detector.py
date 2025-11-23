import difflib

class WakeWordDetector:
    """
    Detecta la palabra clave ('Sanna') con tolerancia fonética controlada.
    """
    def __init__(self, keyword="lucy"):
        self.keyword = keyword.lower()

    def detect(self, text: str) -> bool:
        if not text:
            return False
        text = text.lower().replace("í", "i").replace("í", "i")
        for w in text.split():
            if difflib.SequenceMatcher(None, w, self.keyword).ratio() > 0.82:
                return True
        return False
