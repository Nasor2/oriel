# output/tts_manager.py
import pyttsx3
import threading
import queue
import time
from colorama import Fore

class TTSManager:
    """
    Gestor de síntesis de voz con prioridad, cooldown y sincronización.
    - Suscrito a 'tts_request' y 'tts_request_urgent'.
    - Publica 'tts_start' y 'tts_end' para que el Orchestrator/core sepa cuando hablar.
    """
    PRIORITY_LEVELS = {"urgent": 0, "normal": 1, "low": 2}

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.queue = queue.PriorityQueue()
        self._stop = False
        self._lock = threading.Lock()
        self._last_speak_time = 0.0
        self._last_urgent_message = ""  # Para evitar repetir mensajes urgentes
        self._last_normal_message = ""  # Para evitar repetir mensajes normales

        # Subscripción a eventos de TTS
        self.event_bus.subscribe("tts_request", self._on_tts_request)
        self.event_bus.subscribe("tts_request_urgent", self._on_tts_request)

        # iniciar hilo consumidor
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        print(Fore.GREEN + "[TTSManager] Inicializado y suscrito a eventos de TTS.")

    # ------------------------------------------------------------------
    def _normalize_payload(self, payload):
        """Normaliza payload recibido (string o dict)."""
        if isinstance(payload, str):
            return (self.PRIORITY_LEVELS["normal"], payload)
        if isinstance(payload, dict):
            text = payload.get("text", "")
            priority = payload.get("priority", "normal").lower()
            level = self.PRIORITY_LEVELS.get(priority, 1)
            return (level, text)
        return (self.PRIORITY_LEVELS["normal"], str(payload))

    def _on_tts_request(self, payload):
        level, text = self._normalize_payload(payload)
        if not text or not text.strip():
            return

        # Evitar repetir el mismo mensaje seguido
        if level == 0 and text == self._last_urgent_message:
            print(Fore.LIGHTBLACK_EX + f"[TTSManager] Mensaje urgente repetido ignorado: {text}")
            return
        elif level == 1 and text == self._last_normal_message:
            print(Fore.LIGHTBLACK_EX + f"[TTSManager] Mensaje normal repetido ignorado: {text}")
            return

        # prioridad, timestamp, texto (timestamp para orden FIFO en mismo nivel)
        self.queue.put((level, time.time(), text.strip()))
        color = {0: Fore.RED, 1: Fore.CYAN, 2: Fore.LIGHTBLACK_EX}.get(level, Fore.CYAN)
        tag = {0: "(URGENTE)", 1: "(normal)", 2: "(baja)"}.get(level, "(normal)")
        print(color + f"[TTSManager] {tag} → {text}")

    # ------------------------------------------------------------------
    def _speak_once(self, text: str, priority: int = 1):  # Valor por defecto agregado
        """Habla y notifica start/end (sin solapamientos gracias al lock)."""
        with self._lock:
            try:
                # Guardar el último mensaje según prioridad
                if priority == 0:  # Urgente
                    self._last_urgent_message = text
                elif priority == 1:  # Normal
                    self._last_normal_message = text

                # notificar inicio al bus (para bloquear entrada)
                self.event_bus.publish("tts_start", text)

                engine = pyttsx3.init()
                engine.setProperty("rate", 175)
                for v in engine.getProperty("voices"):
                    if "spanish" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break

                print(Fore.LIGHTYELLOW_EX + f"[TTS] Hablando: {text}")
                engine.say(text)
                engine.runAndWait()
                engine.stop()

                # cooldown breve para evitar que el STT capture el TTS
                self._last_speak_time = time.time()
                time.sleep(0.8)  # Aumentar el tiempo de espera entre mensajes

            except Exception as e:
                print(Fore.RED + f"[TTS ERROR] {e}")
            finally:
                # notificar fin
                self.event_bus.publish("tts_end", text)

    # ------------------------------------------------------------------
    def _worker(self):
        while not self._stop:
            try:
                level, _, text = self.queue.get()
                if text:
                    self._speak_once(text, level)  # Pasar level como priority
            except Exception as e:
                print(Fore.RED + f"[TTSManager] Error en worker: {e}")

    def stop(self):
        self._stop = True
        try:
            self.queue.put((99, time.time(), ""))
        except Exception:
            pass
        self.thread.join(timeout=1)
