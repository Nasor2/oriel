# visual/sai/engine/inference_engine.py
"""
Motor de inferencia YOLOv8 + MiDaS.
Detección de objetos + estimación de profundidad.
Optimizado para CPU.
"""

import torch
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from colorama import Fore
from dataclasses import dataclass

from .depth_estimator import DepthEstimator


@dataclass
class Detection:
    """Estructura de una detección individual."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    depth_estimate: float  # 0.0-1.0


@dataclass
class InferenceResult:
    """Resultado completo de inferencia."""
    detections: List[Detection]
    depth_map: Optional[np.ndarray]
    inference_time_ms: float
    frame_shape: Tuple[int, int, int]


class InferenceEngine:
    """
    Motor de inferencia YOLOv8 + MiDaS.
    Ejecuta detecciones + estimación de profundidad en CPU.
    """

    def __init__(self, yolo_model_path: str,
                 device: str = "cpu",
                 yolo_conf_threshold: float = 0.4,
                 yolo_iou_threshold: float = 0.45,
                 img_size: int = 224,
                 class_names: List[str] = None):
        """
        Inicializa motor de inferencia.

        Args:
            yolo_model_path: ruta al modelo YOLO .pt
            device: "cpu" o "cuda"
            yolo_conf_threshold: umbral de confianza YOLO
            yolo_iou_threshold: umbral NMS
            img_size: tamaño de entrada (debe ser cuadrado)
            class_names: lista de nombres de clases
        """
        self.yolo_model_path = yolo_model_path
        self.device = device
        self.yolo_conf_threshold = yolo_conf_threshold
        self.yolo_iou_threshold = yolo_iou_threshold
        self.img_size = img_size
        self.class_names = class_names or []

        self.yolo_model = None
        self.depth_estimator = None

        self._load_models()

    def _load_models(self):
        """Carga YOLO + MiDaS."""
        # Cargar YOLO
        try:
            print(Fore.CYAN + f"[InferenceEngine] Cargando YOLO desde: {self.yolo_model_path}")
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            self.yolo_model.to(self.device)
            print(Fore.GREEN + "[InferenceEngine] ✅ YOLO cargado correctamente.")
        except Exception as e:
            print(Fore.RED + f"[InferenceEngine] ❌ Error al cargar YOLO: {e}")
            raise

        # Cargar MiDaS
        try:
            print(Fore.CYAN + "[InferenceEngine] Cargando MiDaS...")
            self.depth_estimator = DepthEstimator(
                model_type="DPT_Hybrid",
                device=self.device,
                img_size=self.img_size
            )
            print(Fore.GREEN + "[InferenceEngine] ✅ MiDaS cargado correctamente.")
        except Exception as e:
            print(Fore.YELLOW + f"[InferenceEngine] ⚠️ No se pudo cargar MiDaS: {e}")
            self.depth_estimator = None

    def infer(self, frame: np.ndarray, estimate_depth: bool = True) -> InferenceResult:
        """
        Ejecuta inferencia completa (YOLO + MiDaS).

        Args:
            frame: frame BGR (H, W, 3)
            estimate_depth: si True, estima profundidad con MiDaS

        Returns:
            InferenceResult con detecciones + depth_map
        """
        import time
        t_start = time.time()

        if self.yolo_model is None:
            raise RuntimeError("[InferenceEngine] YOLO no cargado.")

        frame_shape = frame.shape

        # ============================================================
        # 1. INFERENCIA YOLO
        # ============================================================
        results = self.yolo_model.predict(
            source=frame,
            conf=self.yolo_conf_threshold,
            iou=self.yolo_iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            # Extraer boxes
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Coordenadas
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    x1, y1, x2, y2 = xyxy

                    # Confianza
                    conf = float(box.conf[0].cpu().numpy())

                    # Clase
                    cls_id = int(box.cls[0].cpu().numpy())

                    # Validar clase
                    if cls_id >= len(self.class_names):
                        continue

                    cls_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"

                    # Crear detección
                    detection = Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=[x1, y1, x2, y2],
                        depth_estimate=0.5  # Default (se actualiza abajo)
                    )

                    detections.append(detection)

        # ============================================================
        # 2. INFERENCIA MIDAS (estimación de profundidad)
        # ============================================================
        depth_map = None

        if estimate_depth and self.depth_estimator is not None:
            try:
                depth_map = self.depth_estimator.estimate(frame)

                # Actualizar depth_estimate para cada detección
                for detection in detections:
                    depth_val = self.depth_estimator.estimate_bbox_depth(
                        frame, detection.bbox, depth_map
                    )
                    detection.depth_estimate = depth_val
            except Exception as e:
                print(Fore.YELLOW + f"[InferenceEngine] ⚠️ Error en MiDaS: {e}")
                depth_map = None

        # ============================================================
        # 3. RESULTADO FINAL
        # ============================================================
        t_end = time.time()
        inference_time_ms = (t_end - t_start) * 1000.0

        result = InferenceResult(
            detections=detections,
            depth_map=depth_map,
            inference_time_ms=inference_time_ms,
            frame_shape=frame_shape
        )

        return result

    def get_detections_dict(self, inference_result: InferenceResult) -> List[Dict]:
        """
        Convierte InferenceResult a lista de diccionarios.

        Args:
            inference_result: resultado de infer()

        Returns:
            lista de dicts: [{class, confidence, bbox, depth_estimate}, ...]
        """
        return [
            {
                "class_id": det.class_id,
                "class": det.class_name,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "depth_estimate": det.depth_estimate,
            }
            for det in inference_result.detections
        ]

