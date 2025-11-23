# input/audio/audio_pipeline.py
import time
from colorama import Fore, init
from input.audio.audio_manager import AudioManager
from input.audio.wakeword_detector import WakeWordDetector
from input.audio.command_processor import CommandProcessor
from core.contracts.audio_contract import AudioIntentPayload

init(autoreset=True)

class AudioPipeline:
    """
    Pipeline de audio integrado con EventBus / Orchestrator / TTSManager.
    - Detecta wakeword y abre ventana de escucha.
    - Distinguimos wakeword sola vs wakeword+comando.
    - Respeta bloqueo TTS (tts_start / tts_end) para evitar eco y conflictos.
    - Publica intent_detected sólo para intents externos válidos.
    """

    def __init__(self, event_bus, wakeword="lucy", listen_timeout=4.0, min_confidence=0.55):
        self.event_bus = event_bus
        self.audio = AudioManager()
        self.wakeword = WakeWordDetector(keyword=wakeword)
        self.processor = CommandProcessor()

        self.active = False                # True tras wakeword válida y antes de recibir comando
        self.listen_deadline = 0.0
        self.listen_timeout = listen_timeout
        self.min_confidence = min_confidence

        self.tts_blocked = False           # True mientras TTSManager esté hablando
        self.last_user_interaction = 0     # Para controlar el tiempo desde la última interacción
        # Suscribirse para sincronizar con TTS (orquestador también escucha estos eventos)
        self.event_bus.subscribe("tts_start", self._on_tts_start)
        self.event_bus.subscribe("tts_end", self._on_tts_end)

    def _on_tts_start(self, payload):
        # marca bloqueo localmente (el Orchestrator actualizará CoreContext)
        self.tts_blocked = True
        self.last_user_interaction = time.time()  # Registrar inicio de interacción del sistema
        print(Fore.LIGHTBLACK_EX + "[PIPELINE] TTS iniciado -> bloqueo local activado.")

    def _on_tts_end(self, payload):
        self.tts_blocked = False
        self.last_user_interaction = time.time()  # Registrar fin de interacción del sistema
        print(Fore.LIGHTBLACK_EX + "[PIPELINE] TTS finalizado -> bloqueo local desactivado.")

    def run(self):
        """Loop principal (bloqueante)."""
        print(Fore.MAGENTA + "\n [INITIALIZED] ORIEL iniciado. Di 'Lucy' para activar.\n")

        while True:
            # Solo escuchar cuando no hay TTS activo
            if not self.tts_blocked:
                audio = self.audio.listen()
                text = self.audio.recognize(audio)
            else:
                # Si TTS está activo, esperar un poco antes de volver a escuchar
                time.sleep(0.5)
                continue

            # Si no hay texto reconocido...
            if not text:
                # si estábamos activos, comprobar timeout de escucha
                if self.active and time.time() > self.listen_deadline:
                    self.active = False
                    # anunciar modo pasivo (solo si TTS no está bloqueando)
                    if not self.tts_blocked:
                        self.event_bus.publish("tts_request", {"text": "Modo pasivo. Esperando la palabra 'Lucy'.",
                                                               "priority": "normal"})
                    print(Fore.LIGHTBLACK_EX + "[TIMEOUT] Ventana de escucha cerrada por timeout.")
                else:
                    print(Fore.YELLOW + "[INFO] No se detectó voz clara.")
                continue

            print(Fore.CYAN + f"[TEXT] Texto: {text}")

            # Si TTS está hablando, ignoramos la entrada para evitar eco/conflictos
            # Pero permitir comandos de desactivación incluso durante TTS
            current_time = time.time()
            if self.tts_blocked and (current_time - self.last_user_interaction) < 2.0:
                # Solo ignorar si es inmediatamente después de una interacción del sistema
                disable_commands = ["desactiva", "apaga", "detén", "desactivar", "apagar", "detener"]
                if not any(disable_word in text.lower() for disable_word in disable_commands):
                    print(Fore.YELLOW + "[PIPELINE] Ignorado input (TTS en curso).")
                    continue

            txt_clean = text.lower().strip()

            # --- Si no estamos activos (esperando wakeword) ---
            if not self.active:
                if self.wakeword.detect(txt_clean):
                    tokens = txt_clean.split()
                    # si la frase es sólo la wakeword -> abrir ventana de escucha
                    if len(tokens) == 1:
                        self.active = True
                        self.listen_deadline = time.time() + self.listen_timeout
                        self.event_bus.publish("tts_request", {"text": "Sí, te escucho.", "priority": "normal"})
                        print(Fore.GREEN + "[WAKEWORD] Activado. Ventana de escucha abierta.")
                        continue
                    else:
                        # wakeword + comando -> quitar wakeword y procesar el resto
                        remainder = " ".join(tokens[1:]).strip()
                        if not remainder:
                            # raro, tratamos como wakeword sola
                            self.active = True
                            self.listen_deadline = time.time() + self.listen_timeout
                            self.event_bus.publish("tts_request", {"text": "Sí, te escucho.", "priority": "normal"})
                            continue
                        else:
                            text = remainder  # seguir procesando como comando
                            print(
                                Fore.GREEN + "[WAKEWORD] Detectada + comando en la misma frase. Procesando comando...")
                else:
                    print(Fore.YELLOW + "[WAITING] Esperando la palabra clave 'Lucy'...")
                    continue

            # --- En modo activo -> procesar comando recibido ---
            # evitar frases triviales (wakeword reiterada o respuestas del propio TTS)
            trivials = {"lucy", "sí", "si", "te escucho", "ya estoy escuchando", "hola lucy"}
            if txt_clean in trivials:
                print(Fore.LIGHTBLACK_EX + "[INFO] Ignorado texto trivial tras wakeword.")
                self.active = False
                continue

            # NLP: detectar intención
            intent, response, confidence = self.processor.process(text)
            print(Fore.BLUE + f"[DETECT] Intención: {intent} (confianza={confidence:.2f})")

            # Registrar interacción del usuario
            self.last_user_interaction = time.time()

            # si es intent externo y confianza suficiente -> publicar al core
            if intent in self.processor.EXTERNAL_INTENTS and confidence >= self.min_confidence:
                payload = AudioIntentPayload.create(
                    text=text,
                    intent=intent,
                    confidence=round(confidence, 2),
                    meta={"engine": "google_stt"}
                )
                self.event_bus.publish("intent_detected", payload)
                print(Fore.WHITE + f"[PAYLOAD] {payload.to_dict()}")

            elif intent in self.processor.INTERNAL_INTENTS:
                if response:
                    self.event_bus.publish("tts_request", {"text": response, "priority": "normal"})
            else:
                # desconocido o baja confianza
                self.event_bus.publish("tts_request",
                                       {"text": "No entendí lo que quisiste decir.", "priority": "normal"})

            # cerrar ciclo de escucha
            self.active = False
            # ligera pausa para evitar loops rápidos
            time.sleep(0.1)  # Aumentar ligeramente el tiempo de pausa
