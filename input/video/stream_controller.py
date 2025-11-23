import threading
import time
from colorama import Fore


class AlertStreamController:
    """
     Controla la captura continua de frames para el sistema de alertas.
    Diseñado para operar en background y ejecutarse mientras el toggle esté activo.
    """

    def __init__(self, video_manager, callback_fn, target_fps=15):
        """
        :param video_manager: instancia de VideoManager
        :param callback_fn: función que recibe cada payload procesado
        :param target_fps: frames por segundo (recomendado: 10–20)
        """
        self.video_manager = video_manager
        self.callback_fn = callback_fn
        self.target_fps = target_fps

        self.active = False
        self.thread = None
        self._lock = threading.Lock()

        # NUEVO: Frame skipping para reducir spam
        # A 15 FPS, procesar cada 8 frames = ~0.5s entre procesos
        self.frame_skip = 8  # Procesar 1 de cada 8 frames
        self.frame_count = 0

    # -----------------------------
    def start(self):
        """Inicia el stream en background."""
        with self._lock:
            if self.active:
                print(Fore.YELLOW + " Stream de alertas ya está activo.")
                return

            self.active = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print(Fore.GREEN + f" Stream de alertas iniciado ({self.target_fps} FPS máx).")

    # -----------------------------
    def stop(self):
        """Detiene el stream y libera recursos."""
        with self._lock:
            if not self.active:
                print(Fore.YELLOW + " Stream de alertas ya estaba detenido.")
                return

            self.active = False
            print(Fore.RED + " Stream de alertas detenido correctamente.")

    # -----------------------------
    def _run_loop(self):
        frame_interval = 1.0 / self.target_fps
        last_time = time.time()

        while self.active:
            now = time.time()
            if now - last_time < frame_interval:
                time.sleep(0.005)
                continue
            last_time = now

            payload = self.video_manager.capture_frame()
            if not payload:
                continue

            # NUEVO: Frame skipping - procesar solo 1 de cada N frames
            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue  # Saltar este frame

            # Procesar este frame
            try:
                self.callback_fn(payload)
            except Exception as e:
                print(Fore.RED + f" Error en callback de stream: {e}")
                time.sleep(0.2)

        print(Fore.LIGHTBLACK_EX + " Loop de stream de alertas finalizado.")
