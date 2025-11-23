from dataclasses import dataclass, field
import time


@dataclass
class CoreContext:
    """
    Estado global del sistema ORIEL.
    Gestiona el modo de alerta, último intent reconocido,
    y los bloqueos temporales de entrada o recursos (ej. TTS activo).
    """
    alert_system_enabled: bool = False
    listening_mode: bool = False
    last_intent: str = None
    active_locks: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def update_intent(self, intent: str):
        """Actualiza el último intent detectado."""
        self.last_intent = intent

    # ------------------------------------------------------------------
    def lock_input(self, key: str, timeout: float = 3.0):
        """Bloquea temporalmente una entrada o recurso."""
        self.active_locks[key] = time.time() + timeout

    def is_locked(self, key: str) -> bool:
        """Verifica si un recurso o entrada está bloqueado."""
        expiry = self.active_locks.get(key)
        if not expiry:
            return False
        if time.time() > expiry:
            self.active_locks.pop(key, None)
            return False
        return True

    def unlock(self, key: str):
        """Desbloquea manualmente un contexto."""
        self.active_locks.pop(key, None)
