import cv2
import time
from colorama import Fore
from input.video.frame_preprocessor import FramePreprocessor
from core.contracts.video_contract import VisualFramePayload
from input.video.video_profiles import get_default_video_profiles
from input.video.stream_controller import AlertStreamController


class VideoManager:
    """
    🎥 Administra la fuente de video del sistema Zacarías:
    - Soporta cámara local, RTSP, HTTP o archivo.
    - Realiza reconexión automática.
    - Genera payloads estandarizados (VisualFramePayload).
    - Gestiona modo continuo de alertas con control de FPS.
    """

    def __init__(self, source=0, base_resolution=(640, 480), view_profiles=None):
        self.source = source
        self.base_resolution = base_resolution
        self.view_profiles = view_profiles or get_default_video_profiles()
        self.preprocessor = FramePreprocessor(self.view_profiles)
        self.cap = None
        self.alert_stream = None

        self._open_source()

    # -------------------------------
    def _open_source(self):
        print(Fore.CYAN + f"[VideoManager] Abriendo fuente de video: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(Fore.RED + f"[ERROR] No se pudo abrir la fuente: {self.source}")

        # Solo aplica a cámaras locales
        if isinstance(self.source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.base_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.base_resolution[1])

        print(Fore.GREEN + f"[OK] Fuente de video lista ({self._detect_stream_type()}).")

    # -------------------------------
    def capture_frame(self):
        """Captura un solo frame y genera un payload estandarizado."""
        if not self.cap or not self.cap.isOpened():
            print(Fore.YELLOW + "[VideoManager] Fuente desconectada. Reintentando...")
            time.sleep(1)
            self._open_source()
            return None

        ret, frame = self.cap.read()
        if not ret:
            print(Fore.YELLOW + "[VideoManager] No se pudo leer el frame. Esperando...")
            time.sleep(0.3)
            return None

        # Convertir a RGB (requisito estándar para los modelos)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generar vistas procesadas
        processed_views = self.preprocessor.generate_views(frame_rgb)

        # Crear payload
        payload = VisualFramePayload.create_from_frame(
            frame_rgb,
            processed_views,
            metadata={
                "source": str(self.source),
                "resolution": self.base_resolution,
                "stream_type": self._detect_stream_type(),
            }
        )
        return payload

    # -------------------------------
    def _detect_stream_type(self):
        """Determina el tipo de fuente."""
        if isinstance(self.source, int):
            return "local_camera"
        elif str(self.source).startswith("rtsp"):
            return "rtsp_stream"
        elif str(self.source).startswith("http"):
            return "http_stream"
        elif str(self.source).endswith(".mp4"):
            return "video_file"
        else:
            return "unknown"

    # -------------------------------
    def start_alert_stream(self, callback_fn, fps=15):
        """Inicia el modo de stream continuo (ej. sistema de alertas activo)."""
        if self.alert_stream is None:
            self.alert_stream = AlertStreamController(self, callback_fn, fps)
        self.alert_stream.start()

    def stop_alert_stream(self):
        """Detiene el stream continuo de alertas."""
        if self.alert_stream:
            self.alert_stream.stop()

    # -------------------------------
    def release(self):
        """Libera recursos."""
        if self.cap:
            self.cap.release()
            print(Fore.LIGHTBLACK_EX + "[VideoManager] Fuente de video liberada.")
