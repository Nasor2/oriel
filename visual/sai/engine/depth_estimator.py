# visual/sai/engine/depth_estimator.py
"""
Estimador de profundidad usando MiDaS DPT Hybrid.
CPU-optimizado.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional
from colorama import Fore


class DepthEstimator:
    """
    Wrapper para MiDaS DPT Hybrid.
    Estima mapa de profundidad desde frame RGB.
    """

    def __init__(self, model_type: str = "DPT_Hybrid", device: str = "cpu", img_size: int = 224):
        """
        Inicializa estimador de profundidad.

        Args:
            model_type: tipo de modelo MiDaS ("DPT_Hybrid" recomendado)
            device: "cpu" o "cuda"
            img_size: tamaño de entrada del modelo
        """
        self.model_type = model_type
        self.device = device
        self.img_size = img_size
        self.model = None

        self._load_model()

    def _load_model(self):
        """Carga modelo MiDaS DPT Hybrid desde archivo local (dpt_hybrid_384.pt) sin descargas."""
        try:
            import os

            print(Fore.CYAN + f"[DepthEstimator] Cargando MiDaS ({self.model_type})...")

            # Rutas donde buscar el archivo descargado
            checkpoint_paths = [
                "models/dpt_hybrid_384.pt",
                "dpt_hybrid_384.pt",
                os.path.expanduser("~/.cache/torch/hub/checkpoints/dpt_hybrid_384.pt")
            ]

            checkpoint_path = None
            for path in checkpoint_paths:
                if os.path.isfile(path):
                    checkpoint_path = path
                    break

            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"❌ No se encontró dpt_hybrid_384.pt en: {checkpoint_paths}\n"
                    f"Descargalo desde: https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"
                )

            # Intentar cargar con midas-pytorch si está disponible
            try:
                from midas.dpt_depth import DPTDepthModel

                self.model = DPTDepthModel(
                    path_backbone=None,
                    non_negative=True,
                    enable_attention_hooks=False,
                )

                # Cargar pesos desde archivo local
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)

            except ImportError:
                # Fallback: cargar directamente con torch (como modelo serializado)
                print(Fore.YELLOW + "[DepthEstimator] midas-pytorch no disponible, intentando carga directa...")

                # Intentar cargar el checkpoint como modelo genérico
                try:
                    # El checkpoint puede contener estado o ser un modelo TorchScript
                    checkpoint = torch.load(checkpoint_path, map_location=self.device)

                    # Si es un dict de state_dict, necesitamos arquitectura
                    # Intentamos usar PyTorch Hub como fallback (una sola descarga)
                    print(Fore.LIGHTBLACK_EX + "[DepthEstimator] Cargando arquitectura desde Hub (una sola vez)...")
                    self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid",
                                                trust_repo=True, force_reload=False)

                    # Si cargó algo, inyectar pesos locales
                    if isinstance(checkpoint, dict):
                        self.model.load_state_dict(checkpoint, strict=False)
                    else:
                        self.model = checkpoint

                except Exception as hub_err:
                    print(Fore.RED + f"[DepthEstimator] Incluso Hub falló: {hub_err}")
                    raise

            self.model.to(self.device)
            self.model.eval()
            print(Fore.GREEN + f"[DepthEstimator] MiDaS cargado desde: {checkpoint_path}")

        except FileNotFoundError as fe:
            print(Fore.RED + f"[DepthEstimator] {fe}")
            raise
        except Exception as e:
            print(Fore.RED + f"[DepthEstimator] Error al cargar MiDaS: {e}")
            raise

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estima mapa de profundidad del frame.

        Args:
            frame: frame BGR (H, W, 3)

        Returns:
            depth_map normalizado (0.0-1.0) con shape (img_size, img_size)
        """
        if self.model is None:
            raise RuntimeError("[DepthEstimator] Modelo no cargado.")

        # Convertir BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Redimensionar a tamaño del modelo
        rgb_resized = cv2.resize(rgb, (self.img_size, self.img_size))

        # Normalizar a [0, 1]
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0

        # Convertir a tensor: (H, W, 3) -> (1, 3, H, W)
        tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Inferencia
        with torch.no_grad():
            depth_raw = self.model(tensor)

            # Interpolar a tamaño del modelo
            depth_interp = F.interpolate(
                depth_raw.unsqueeze(1),
                size=(self.img_size, self.img_size),
                mode="bicubic",
                align_corners=False
            ).squeeze()

        # Convertir a numpy y normalizar
        depth_np = depth_interp.cpu().numpy()

        # Normalizar a [0, 1]
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        if depth_max - depth_min > 0:
            depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.ones_like(depth_np) * 0.5

        return depth_normalized.astype(np.float32)

    def estimate_bbox_depth(self, frame: np.ndarray, bbox: list,
                           depth_map: Optional[np.ndarray] = None) -> float:
        """
        Estima profundidad promedio dentro de un bbox.

        Args:
            frame: frame original (para inferencia si depth_map es None)
            bbox: [x1, y1, x2, y2] en coordenadas del frame
            depth_map: mapa de profundidad precomputado (opcional)

        Returns:
            valor de profundidad promedio (0.0-1.0)
        """
        if depth_map is None:
            depth_map = self.estimate(frame)

        # Escalar bbox al tamaño del depth_map
        h_orig, w_orig = frame.shape[:2]
        h_depth, w_depth = depth_map.shape

        scale_x = w_depth / w_orig
        scale_y = h_depth / h_orig

        x1, y1, x2, y2 = bbox
        x1_scaled = int(max(0, x1 * scale_x))
        y1_scaled = int(max(0, y1 * scale_y))
        x2_scaled = int(min(w_depth, x2 * scale_x))
        y2_scaled = int(min(h_depth, y2 * scale_y))

        # Extraer región
        roi = depth_map[y1_scaled:y2_scaled, x1_scaled:x2_scaled]

        if roi.size == 0:
            return 0.5  # Default

        # Retornar promedio
        return float(np.mean(roi))

