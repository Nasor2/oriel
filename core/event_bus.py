import threading
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class EventBus:
    """
    Sistema de publicación/suscripción simple (thread-safe).
    Permite comunicación asíncrona entre los módulos del sistema ORIEL.
    """

    def __init__(self):
        self.subscribers = defaultdict(list)
        self._lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=5)

    def subscribe(self, event_name: str, callback):
        """Registra una función que será llamada cuando ocurra el evento."""
        with self._lock:
            self.subscribers[event_name].append(callback)

    def publish(self, event_name: str, payload=None):
        callbacks = self.subscribers.get(event_name, [])
        for cb in callbacks:
            # Enviar la tarea al pool en lugar de crear un hilo nuevo
            self.executor.submit(cb, payload)

    def shutdown(self):
        """Cierra el pool de hilos correctamente."""
        print("Apagando EventBus...")
        self.executor.shutdown(wait=False)
