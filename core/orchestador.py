# core/orchestrator.py

from colorama import Fore
from core.core_context import CoreContext
from visual.footpath.footpath_system import FootpathSystem

class Orchestrator:
    """
    Núcleo central del sistema ORIEL.
    Gestiona intenciones, prioridad de salida TTS y sincronización del estado global.
    """

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.ctx = CoreContext()

        # Suscripciones principales
        self.event_bus.subscribe("intent_detected", self.on_intent)
        self.event_bus.subscribe("frame_ready", self.on_frame)
        # Suscribir alertas visuales publicadas por AlertSystem
        self.event_bus.subscribe("visual_alert", self.on_alert)

        # TTS bloqueo sincronizado
        self.event_bus.subscribe("tts_start", self._on_tts_start)
        self.event_bus.subscribe("tts_end", self._on_tts_end)

        # Estado interno
        self.tts_busy = False
        self.last_intent = None
        self.last_alert_message = ""  # Para evitar repetir alertas

        self.alert_system = None
        self.footpath_system = None
        self.video_manager = None

    def set_alert_system(self, alert_system):
        """Registrar instancia de AlertSystem para que el orquestador pueda controlarla."""
        self.alert_system = alert_system
        print(Fore.LIGHTGREEN_EX + "[Orchestrator] AlertSystem registrado.")

    # ------------------------------------------------------------------
    # BLOQUEO DE INPUT DURANTE TTS
    def _on_tts_start(self, text):
        if not self.tts_busy:
            self.tts_busy = True
            self.ctx.lock_input("tts_playing", timeout=5.0)
            print(Fore.LIGHTBLACK_EX + "[CORE] Entrada bloqueada (TTS en curso).")

    def _on_tts_end(self, text):
        self.tts_busy = False
        self.ctx.unlock("tts_playing")
        print(Fore.LIGHTBLACK_EX + "[CORE] Entrada desbloqueada (TTS finalizado).")

    def set_footpath_system(self, footpath_system):
        """Registrar instancia de FootpathSystem."""
        self.footpath_system = footpath_system
        # Suscribirse a eventos de footpath
        self.event_bus.subscribe("footpath_result", self.on_footpath_result)
        print(Fore.LIGHTGREEN_EX + "[Orchestrator] FootpathSystem registrado.")

    def set_video_manager(self, video_manager):
        """Registrar instancia de VideoManager."""
        self.video_manager = video_manager
        print(Fore.LIGHTGREEN_EX + "[Orchestrator] VideoManager registrado.")

    # ------------------------------------------------------------------
    # 🗣️ PUBLICACIÓN CENTRAL DE TTS
    def _publish_tts(self, text: str, urgent: bool = False):
        if not text:
            return

        payload = {
            "text": text,
            "priority": "urgent" if urgent else "normal",
            "source": "orchestrator"
        }

        event = "tts_request_urgent" if urgent else "tts_request"
        self.event_bus.publish(event, payload)

        color = Fore.RED if urgent else Fore.CYAN
        tag = "(URGENTE)" if urgent else "(normal)"

        print(color + f"[Orchestrator] Emite TTS {tag}: {text}")

    # ------------------------------------------------------------------
    # PROCESAMIENTO DE INTENTS
    def on_intent(self, payload):
        if self.ctx.is_locked("tts_playing"):
            print(Fore.LIGHTBLACK_EX + "[CORE] Ignorado intent (TTS en curso).")
            return

        intent = getattr(payload, "intent", None)
        confidence = getattr(payload, "confidence", 0.0)

        if not intent:
            self._publish_tts("No entendí el comando.")
            return

        # Evitar repetición
        if intent == self.last_intent and confidence < 1.0:
            print(Fore.LIGHTBLACK_EX + f"[CORE] Intento repetido ignorado: {intent}")
            return

        self.last_intent = intent
        self.ctx.update_intent(intent)

        print(Fore.CYAN + f"[CORE] Intención recibida: {intent} (conf={confidence:.2f})")

        # ===================================================
        # CONTROL DEL SISTEMA DE ALERTAS
        # ===================================================
        if intent == "enable_alert_system":
            if not self.ctx.alert_system_enabled:
                self.ctx.alert_system_enabled = True
                self.event_bus.publish("sai_command", "enable")
                # if an AlertSystem instance is registered, start it
                if self.alert_system is not None:
                    try:
                        self.alert_system.start()
                        print(Fore.GREEN + "[Orchestrator] AlertSystem arrancado por comando.")
                    except Exception as e:
                        print(Fore.RED + f"[Orchestrator] Error al iniciar AlertSystem: {e}")
                self._publish_tts("Sistema de alertas activado.", urgent=True)
            else:
                self._publish_tts("El sistema de alertas ya está activo.")

        elif intent == "disable_alert_system":
            if self.ctx.alert_system_enabled:
                self.ctx.alert_system_enabled = False
                self.event_bus.publish("sai_command", "disable")
                # if registered, stop it
                if self.alert_system is not None:
                    try:
                        self.alert_system.stop()
                        print(Fore.YELLOW + "[Orchestrator] AlertSystem detenido por comando.")
                    except Exception as e:
                        print(Fore.RED + f"[Orchestrator] Error al detener AlertSystem: {e}")
                self._publish_tts("Sistema de alertas desactivado.", urgent=True)
            else:
                self._publish_tts("El sistema de alertas ya estaba desactivado.")

        # ===================================================
        # OTROS INTENTS
        # ===================================================
        elif intent == "scene_description":
            self._publish_tts("Procesando la escena frente a ti...")

        elif intent == "footpath_request":
            self._publish_tts("Analizando el camino frente a ti.")
            # Activar y procesar solicitud de footpath
            if self.footpath_system is not None:
                try:
                    self.footpath_system.activate()
                    self.footpath_system.process_footpath_request(
                        self.video_manager,
                        alert_system_enabled=self.ctx.alert_system_enabled
                    )
                except Exception as e:
                    print(Fore.RED + f"[Orchestrator] Error procesando solicitud de footpath: {e}")
                    self._publish_tts("Error al analizar el camino. Por favor, intente nuevamente.")
        elif intent == "unknown" or confidence < 0.5:
                self._publish_tts("No entendí bien el comando.")
        else:
            self._publish_tts(f"Ejecutando acción: {intent}.")

    # ------------------------------------------------------------------
    # FRAME HANDLING (solo si alertas activas)
    def on_frame(self, payload):
        if not self.ctx.alert_system_enabled:
            return

        desc = payload.describe()
        print(Fore.GREEN + f"[CORE] Nuevo frame procesado: {desc}")

    # core/orchestrador.py
    def on_alert(self, alert_payload):
        """
        Recibe VisualAlertEvent y lo transforma en TTS con control mejorado de repetición.
        """
        if not self.ctx.alert_system_enabled and self.footpath_system:
            try:
                self.footpath_system.process_pending_requests(self.video_manager)
            except Exception as e:
                print(Fore.RED + f"[Orchestrator] Error procesando solicitudes pendientes: {e}")

        try:
            # Extraer mensaje y nivel
            msg = getattr(alert_payload, "message", None) or getattr(alert_payload, "text", None) or (
                alert_payload.get("message") if isinstance(alert_payload, dict) else None)
            level = getattr(alert_payload, "level", "medium") if not isinstance(alert_payload,
                                                                                dict) else alert_payload.get("level",
                                                                                                             "medium")

            if msg:
                # Control de repetición más robusto usando hash del contenido
                import time
                import hashlib
                current_time = time.time()

                # Crear identificador único del contenido de la alerta
                msg_content = f"{msg}_{level}"
                msg_hash = hashlib.md5(msg_content.encode()).hexdigest()[:16]  # 16 caracteres

                # Verificar si es una alerta repetida reciente (aumentado a 5 segundos)
                if hasattr(self, '_recent_alert_hashes'):
                    if msg_hash in self._recent_alert_hashes:
                        last_time, last_msg = self._recent_alert_hashes[msg_hash]
                        # Aumentar tiempo de espera para evitar repetición
                        if (current_time - last_time) < 5.0:
                            print(Fore.LIGHTBLACK_EX + f"[ALERT] Alerta repetida ignorada: {msg}")
                            return
                else:
                    self._recent_alert_hashes = {}

                # Registrar esta alerta
                self._recent_alert_hashes[msg_hash] = (current_time, msg)

                # Limpiar hashes antiguos (más de 30 segundos)
                expired_hashes = [k for k, v in self._recent_alert_hashes.items()
                                  if current_time - v[0] > 30.0]
                for h in expired_hashes:
                    del self._recent_alert_hashes[h]

                urgent = True if level == "high" else False
                self._publish_tts(msg, urgent=urgent)

                # LOG para debugging
                print(Fore.LIGHTBLUE_EX + f"[ALERT PROCESSED] level={level}, msg='{msg}', urgent={urgent}")
            else:
                print(Fore.YELLOW + "[ALERT] Alerta sin mensaje")
        except Exception as e:
            print(Fore.RED + f"[CORE] Error en on_alert: {e}")
            import traceback
            traceback.print_exc()



    def on_footpath_result(self, footpath_payload):
        """
        Recibe VisualFootpathEvent y lo transforma en TTS.
        """
        try:
            if footpath_payload.message:
                # Publicar mensaje por TTS
                urgent = footpath_payload.risk_level == "high"
                self._publish_tts(footpath_payload.message, urgent=urgent)

                # Log para debugging
                print(Fore.LIGHTBLUE_EX + f"[FOOTPATH] risk={footpath_payload.risk_level}, "
                                          f"action={footpath_payload.action}, msg='{footpath_payload.message}'")
            else:
                print(Fore.YELLOW + "[FOOTPATH] Resultado sin mensaje")
        except Exception as e:
            print(Fore.RED + f"[Orchestrator] Error en on_footpath_result: {e}")
            import traceback
            traceback.print_exc()

