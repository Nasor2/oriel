# visual/sai/pipeline/object_tracker.py
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class TrackedObject:
    """Representa un objeto con seguimiento temporal"""
    track_id: str
    class_name: str
    first_seen: float
    last_seen: float
    last_alerted: Dict[str, float] = field(default_factory=dict)  # nivel_riesgo: timestamp
    positions: deque = field(default_factory=lambda: deque(maxlen=10))  # Historial de posiciones
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=5))  # Historial de bboxes
    risk_history: List[Dict] = field(default_factory=list)  # Historial de riesgos

    def update_position(self, bbox: List[int], timestamp: float):
        """Actualizar posición y bbox"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.positions.append((cx, cy, timestamp))
        self.bbox_history.append(bbox)
        self.last_seen = timestamp

    def get_velocity(self) -> float:
        """Calcular velocidad en píxeles por segundo"""
        if len(self.positions) < 2:
            return 0.0

        # Últimas dos posiciones
        pos2, time2 = self.positions[-1][:2], self.positions[-1][2]
        pos1, time1 = self.positions[-2][:2], self.positions[-2][2]

        if time2 - time1 <= 0:
            return 0.0

        distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        return distance / (time2 - time1)

    def is_moving_significantly(self, threshold: float = 50.0) -> bool:
        """Verificar si el objeto se mueve significativamente"""
        return self.get_velocity() > threshold

    def should_alert(self, risk_level: str, current_time: float,
                     min_interval: float = 3.0) -> bool:
        """Determinar si se debe alertar según nivel de riesgo y tiempo"""
        last_alert_time = self.last_alerted.get(risk_level, 0.0)
        return (current_time - last_alert_time) >= min_interval

    def mark_alerted(self, risk_level: str, timestamp: float):
        """Marcar que se ha alertado en este nivel de riesgo"""
        self.last_alerted[risk_level] = timestamp

class ObjectTracker:
    """Sistema avanzado de seguimiento de objetos"""

    def __init__(self):
        self.tracked_objects: Dict[str, TrackedObject] = {}
        self.next_track_id = 0
        self.class_counters: Dict[str, int] = defaultdict(int)

    def _generate_track_id(self, class_name: str) -> str:
        """Generar ID único para seguimiento"""
        self.class_counters[class_name] += 1
        track_id = f"{class_name}_{self.class_counters[class_name]}_{int(time.time())}"
        return track_id

    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcular IoU entre dos bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersección
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Áreas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _match_detection_to_track(self, detection: Dict, timestamp: float,
                                 max_distance: float = 100.0) -> Optional[str]:
        """Encontrar track existente para una detección"""
        if not detection.get('bbox'):
            return None

        current_bbox = detection['bbox']
        current_class = detection['class']
        current_cx = (current_bbox[0] + current_bbox[2]) / 2
        current_cy = (current_bbox[1] + current_bbox[3]) / 2

        best_match_id = None
        best_score = 0.0

        # Buscar entre objetos existentes de la misma clase
        for track_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.class_name != current_class:
                continue

            # Verificar si está activo (visto en los últimos 2 segundos)
            if timestamp - tracked_obj.last_seen > 2.0:
                continue

            # Calcular distancia al último punto
            if tracked_obj.positions:
                last_pos = tracked_obj.positions[-1]
                distance = np.sqrt((current_cx - last_pos[0])**2 + (current_cy - last_pos[1])**2)

                # También considerar IoU si hay historial de bbox
                iou_score = 0.0
                if tracked_obj.bbox_history:
                    iou_score = self._calculate_iou(current_bbox, tracked_obj.bbox_history[-1])

                # Combinar métricas
                combined_score = (1.0 - min(1.0, distance/max_distance)) * 0.7 + iou_score * 0.3

                if combined_score > best_score and combined_score > 0.3:
                    best_score = combined_score
                    best_match_id = track_id

        return best_match_id

    def update(self, detections: List[Dict], timestamp: float) -> List[Tuple[Dict, str]]:
        """Actualizar seguimiento con nuevas detecciones"""
        updated_detections = []

        # Primero, asociar detecciones existentes
        unmatched_detections = []
        matched_tracks = set()

        for detection in detections:
            track_id = self._match_detection_to_track(detection, timestamp)

            if track_id and track_id in self.tracked_objects:
                # Actualizar track existente
                tracked_obj = self.tracked_objects[track_id]
                tracked_obj.update_position(detection['bbox'], timestamp)
                updated_detections.append((detection, track_id))
                matched_tracks.add(track_id)
            else:
                # Crear nuevo track
                track_id = self._generate_track_id(detection['class'])
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    class_name=detection['class'],
                    first_seen=timestamp,
                    last_seen=timestamp
                )
                tracked_obj.update_position(detection['bbox'], timestamp)
                self.tracked_objects[track_id] = tracked_obj
                updated_detections.append((detection, track_id))

        # Limpiar tracks no vistos por mucho tiempo
        expired_tracks = []
        for track_id, tracked_obj in self.tracked_objects.items():
            if timestamp - tracked_obj.last_seen > 5.0:  # 5 segundos sin ver
                expired_tracks.append(track_id)

        for track_id in expired_tracks:
            del self.tracked_objects[track_id]

        return updated_detections

    def get_tracked_object(self, track_id: str) -> Optional[TrackedObject]:
        """Obtener objeto rastreado por ID"""
        return self.tracked_objects.get(track_id)

    def get_all_tracks(self) -> Dict[str, TrackedObject]:
        """Obtener todos los objetos rastreados"""
        return self.tracked_objects.copy()
