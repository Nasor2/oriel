# visual/sai/pipeline/frame_buffer.py
"""
Buffer circular de frames para tracking temporal.
Mantiene histórico de últimos N frames + detecciones.
"""

import numpy as np
from collections import deque
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class BufferEntry:
    """Entrada en el buffer circular."""
    frame_id: str
    timestamp: float
    detections: List[Dict] = field(default_factory=list)
    depth_map: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


class FrameBuffer:
    """
    Buffer circular de frames.
    Mantiene últimos N frames + detecciones para tracking temporal.
    """

    def __init__(self, capacity: int = 60):
        """
        Inicializa buffer circular.

        Args:
            capacity: número máximo de frames a almacenar
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self._id_to_buffer_idx = {}  # Mapeo: frame_id -> índice

    def add_frame(self, frame_id: str, detections: List[Dict],
                 depth_map: Optional[np.ndarray] = None,
                 metadata: Optional[Dict] = None) -> None:
        """
        Añade un frame al buffer.

        Args:
            frame_id: identificador único del frame
            detections: lista de detecciones [{class, confidence, bbox, depth_estimate}, ...]
            depth_map: mapa de profundidad (opcional)
            metadata: metadatos adicionales {frame_idx, fps, etc}
        """
        entry = BufferEntry(
            frame_id=frame_id,
            timestamp=time.time(),
            detections=detections,
            depth_map=depth_map,
            metadata=metadata or {}
        )

        self.buffer.append(entry)
        self._id_to_buffer_idx[frame_id] = len(self.buffer) - 1

    def get_last_n_frames(self, n: int) -> List[BufferEntry]:
        """
        Retorna últimos N frames.

        Args:
            n: número de frames

        Returns:
            lista de BufferEntry (más recientes al final)
        """
        return list(self.buffer)[-n:]

    def get_all_frames(self) -> List[BufferEntry]:
        """Retorna todos los frames en el buffer."""
        return list(self.buffer)

    def get_frame(self, idx: int) -> Optional[BufferEntry]:
        """
        Obtiene frame por índice (relativo al inicio del buffer).

        Args:
            idx: índice (0 = más antiguo, len-1 = más reciente)

        Returns:
            BufferEntry o None
        """
        if idx < 0 or idx >= len(self.buffer):
            return None
        return list(self.buffer)[idx]

    def get_recent_detections(self, class_name: str,
                             lookback_frames: int = 10) -> List[Tuple[str, Dict]]:
        """
        Obtiene detecciones recientes de una clase específica.

        Args:
            class_name: nombre de la clase
            lookback_frames: cuántos frames atrás buscar

        Returns:
            lista de (frame_id, detection)
        """
        result = []
        recent_frames = self.get_last_n_frames(lookback_frames)

        for entry in recent_frames:
            for det in entry.detections:
                if det.get("class") == class_name:
                    result.append((entry.frame_id, det))

        return result

    def get_temporal_trajectory(self, class_name: str,
                               lookback_frames: int = 5) -> List[Tuple[float, float]]:
        """
        Obtiene trayectoria de un objeto (centroides en últimos N frames).

        Args:
            class_name: nombre de la clase
            lookback_frames: frames a considerar

        Returns:
            lista de (cx, cy) centroides
        """
        trajectory = []
        recent_dets = self.get_recent_detections(class_name, lookback_frames)

        for frame_id, det in recent_dets:
            bbox = det.get("bbox")
            if bbox:
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                trajectory.append((cx, cy))

        return trajectory

    def get_class_count(self, class_name: str,
                       lookback_frames: int = 5) -> int:
        """
        Cuenta cuántas detecciones de una clase hay en últimos N frames.

        Args:
            class_name: nombre de la clase
            lookback_frames: frames a considerar

        Returns:
            conteo
        """
        count = 0
        recent_dets = self.get_recent_detections(class_name, lookback_frames)

        # Contar detecciones únicas (por bbox aproximado)
        unique_bboxes = set()
        for frame_id, det in recent_dets:
            bbox = tuple(det.get("bbox", []))
            unique_bboxes.add(bbox)

        return len(unique_bboxes)

    def get_max_confidence(self, class_name: str,
                          lookback_frames: int = 5) -> float:
        """
        Obtiene confianza máxima de detecciones recientes.

        Args:
            class_name: nombre de la clase
            lookback_frames: frames a considerar

        Returns:
            confianza máxima (0.0-1.0)
        """
        recent_dets = self.get_recent_detections(class_name, lookback_frames)

        if not recent_dets:
            return 0.0

        max_conf = max(det.get("confidence", 0.0) for _, det in recent_dets)
        return max_conf

    def get_average_depth(self, class_name: str,
                         lookback_frames: int = 5) -> float:
        """
        Obtiene profundidad promedio de detecciones recientes.

        Args:
            class_name: nombre de la clase
            lookback_frames: frames a considerar

        Returns:
            profundidad promedio (0.0-1.0)
        """
        recent_dets = self.get_recent_detections(class_name, lookback_frames)

        if not recent_dets:
            return 0.5

        depths = [det.get("depth_estimate", 0.5) for _, det in recent_dets]
        return np.mean(depths)

    def clear(self) -> None:
        """Limpia el buffer."""
        self.buffer.clear()
        self._id_to_buffer_idx.clear()

    def size(self) -> int:
        """Retorna cantidad de frames en el buffer."""
        return len(self.buffer)

    def is_full(self) -> bool:
        """Retorna True si el buffer está lleno."""
        return len(self.buffer) == self.capacity

