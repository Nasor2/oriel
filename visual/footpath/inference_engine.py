# visual/footpath/inference_engine.py
"""
Motor de inferencia para el subsistema Footpath.
Procesa frames y genera máscaras de caminería.
"""

import numpy as np
import cv2
from colorama import Fore
from .model_loader import FootpathModelLoader
from .config import MODEL_CONFIG

class FootpathInferenceEngine:
    """
    Motor de inferencia para análisis de caminerías.
    """

    def __init__(self):
        self.model_loader = FootpathModelLoader()
        self.input_size = MODEL_CONFIG["input_size"]
        print(Fore.CYAN + "[FootpathInference] Motor de inferencia inicializado.")

    def preprocess_frame(self, frame):
        # 1. Redimensionar
        resized = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)

        # 2. Convertir a Escala de Grises (1 Canal)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized

        if gray.ndim > 2:
            gray = gray[..., 0]

        normalized = gray.astype(np.float32)  # Forma: (H, W), Rango: [~0.0, ~1.0]

        processed_channel = np.expand_dims(normalized, axis=-1)  # Forma: (H, W, 1)

        processed = np.expand_dims(processed_channel, axis=0)  # Forma final: (1, H, W, 1)

        print(
            f"Output Shape: {processed.shape}, Dtype: {processed.dtype}, Min: {processed.min()}, Max: {processed.max()}")

        return processed

    def infer(self, frame):
        """
        Realiza inferencia completa en un frame.

        Args:
            frame: numpy array RGB (H, W, 3)

        Returns:
            mask: numpy array (256, 256) con probabilidades de caminería
        """
        try:
            # Preprocesar frame
            processed_frame = self.preprocess_frame(frame)

            # Realizar inferencia
            mask = self.model_loader.predict(processed_frame)

            return mask
        except Exception as e:
            print(Fore.RED + f"[FootpathInference] Error en inferencia: {e}")
            raise
