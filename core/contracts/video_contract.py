from dataclasses import dataclass, field
import numpy as np
import time
from colorama import Fore
import uuid

@dataclass
class VisualFramePayload:
    """
        Contrato estandarizado entre el Input Layer y las Visual Layers.

       Cada payload representa un frame listo para consumo por:
           - Alert System (detección de objetos, profundidad)
           - Footpath Suggestion
           - Scene VQA u otros módulos visuales

       Contiene el frame original, vistas preprocesadas y metadatos del entorno.
       """

    frame_id: str
    timestamp: float
    original_frame: np.ndarray
    processed_views: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def create_from_frame(frame: np.ndarray, processed_views: dict[str, np.ndarray], metadata: dict = None):
        """
        Crea una instancia nueva del payload con un ID y timestamp únicos.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            print(Fore.RED + "[VisualFramePayload] Frame inválido recibido.")
            return None

        return VisualFramePayload(
            frame_id=str(uuid.uuid4()),
            timestamp=time.time(),
            original_frame=frame,
            processed_views=processed_views or {},
            metadata=metadata or {}
        )

    def get_view(self, key: str, default=None):
        """
        Recupera una versión específica del frame (por ejemplo 'yolo' o 'midas').
        """
        return self.processed_views.get(key, default)

    def describe(self) -> str:
        """
        Devuelve una descripción textual resumida del contenido del payload.
        Útil para logs, depuración o monitoreo del orquestador.
        """
        views = list(self.processed_views.keys())
        meta_summary = (
            f"src={self.metadata.get('source', 'N/A')}, "
            f"res={self.metadata.get('resolution', 'N/A')}"
        )
        return (
            f"[FrameID: {self.frame_id[:8]}] "
            f"Views={views} | {meta_summary} | "
            f"t={self.timestamp:.2f}"
        )
    def summary(self) -> str:
        """
        Alias de describe() para mantener compatibilidad con VideoManager.
        """
        return self.describe()
