import cv2
import numpy as np
from colorama import Fore


class FramePreprocessor:
    """
    FramePreprocessor
    Genera versiones adaptadas de un mismo frame para distintos modelos visuales.

    - Escala y normaliza los frames según el perfil de cada modelo.
    - Entrega un diccionario {'yolo': frame_yolo, 'midas': frame_midas, ...}
      que luego se incluye en el VisualFramePayload.
    """

    def __init__(self, view_profiles: dict):
        self.view_profiles = view_profiles or {}
        print(Fore.LIGHTCYAN_EX + f"[FramePreprocessor] inicializado con vistas: {list(self.view_profiles.keys())}")

    # --------------------------------------
    def generate_views(self, frame: np.ndarray) -> dict:
        """
        Genera los frames preprocesados según los perfiles definidos.
        """
        processed_views = {}
        for key, size in self.view_profiles.items():
            try:
                resized = self._resize_frame(frame, size)
                normalized = self._normalize_frame(resized, key)
                processed_views[key] = normalized
            except Exception as e:
                print(Fore.RED + f"[ERROR] al procesar vista '{key}': {e}")
        return processed_views

    # --------------------------------------
    def _resize_frame(self, frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """Redimensiona el frame a la resolución objetivo."""
        width, height = target_size
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    # --------------------------------------
    def _normalize_frame(self, frame: np.ndarray, model_name: str) -> np.ndarray:
        """
        Normaliza los valores de pixel según el modelo destino.
        """
        frame = frame.astype(np.float32)

        if model_name.lower() in ["yolo", "object_detection"]:
            # Normalización estándar [0,1]
            return frame / 255.0

        elif model_name.lower() in ["midas", "depth_estimation"]:
            # MiDaS trabaja con [0,1]
            return frame / 255.0

        elif model_name.lower() in ["footpath", "segmentation"]:
            # Modelos de segmentación: centrado [-1, 1]
            frame = frame / 255.0
            frame -= 0.5
            frame /= 0.5
            return frame

        elif model_name.lower() in ["scene_vqa", "vlm"]:
            # Modelos multimodales (como CLIP o BLIP)
            frame = cv2.resize(frame, (448, 448))
            return frame / 255.0

        else:
            # Default: normalización básica
            return frame / 255.0
