# visual/sai/pipeline/temporal_tracker.py
import time
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

@dataclass
class TemporalObject:
    """Representa un objeto con seguimiento temporal completo"""
    track_id: str
    class_name: str
    first_seen: float
    last_seen: float
    positions: deque = field(default_factory=lambda: deque(maxlen=30))  # Historial extendido
    velocities: deque = field(default_factory=lambda: deque(maxlen=15))  # Historial de velocidades
    distances: deque = field(default_factory=lambda: deque(maxlen=15))   # Historial de distancias
    risk_levels: deque = field(default_factory=lambda: deque(maxlen=8))  # Historial de riesgos
    last_alerted: Dict[str, float] = field(default_factory=dict)  # Por tipo de alerta
    confidence: float = 0.5  # Nivel de confianza de la detección

    def update_observation(self, bbox: List[int], distance: float, risk_level: str,
                          frame_width: int, frame_height: int, timestamp: float, confidence: float = 0.5):
        """Actualizar observación del objeto"""
        # Posición
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        self.positions.append((cx, cy, timestamp))
        self.distances.append(distance)
        self.risk_levels.append(risk_level)
        self.last_seen = timestamp
        self.confidence = confidence

        # Calcular y almacenar velocidad
        if len(self.positions) >= 2:
            pos2, time2 = self.positions[-1][:2], self.positions[-1][2]
            pos1, time1 = self.positions[-2][:2], self.positions[-2][2]
            if time2 - time1 > 0:
                distance_px = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
                velocity = distance_px / (time2 - time1)  # píxeles/segundo
                self.velocities.append(velocity)

    def get_trend_analysis(self) -> Dict:
        """Analizar tendencias del objeto"""
        # Inicializar valores por defecto
        distance_trend = "stable"
        risk_trend = "stable"
        average_distance = 5.0  # Distancia por defecto
        average_velocity = 0.0  # Velocidad por defecto

        if len(self.distances) >= 4:  # Aumentado umbral
            # Análisis de distancia (acercamiento/alejamiento)
            recent_distances = list(self.distances)[-10:]  # Aumentado ventana de análisis
            if len(recent_distances) >= 5:  # Aumentado umbral
                distance_diff = recent_distances[-1] - recent_distances[0]
                if distance_diff < -1.0:  # Aumentado umbral para acercamiento
                    distance_trend = "approaching"
                elif distance_diff > 1.0:  # Aumentado umbral para alejamiento
                    distance_trend = "receding"
                else:
                    distance_trend = "stable"

            # Calcular distancia promedio
            average_distance = np.mean(recent_distances) if recent_distances else 5.0

        # Análisis de riesgo (aumentando/disminuyendo)
        if len(self.risk_levels) >= 4:  # Aumentado umbral
            risk_values = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
            recent_risks = list(self.risk_levels)[-6:]  # Aumentado ventana de análisis
            if len(recent_risks) >= 3:  # Aumentado umbral
                first_risk = risk_values.get(recent_risks[0], 1)
                last_risk = risk_values.get(recent_risks[-1], 1)
                if last_risk > first_risk:
                    risk_trend = "increasing"
                elif last_risk < first_risk:
                    risk_trend = "decreasing"
                else:
                    risk_trend = "stable"

        # Calcular velocidad promedio
        if len(self.velocities) >= 3:  # Aumentado umbral
            average_velocity = np.mean(list(self.velocities)[-7:])  # Aumentado ventana de análisis

        return {
            "distance_trend": distance_trend,
            "risk_trend": risk_trend,
            "average_distance": average_distance,
            "average_velocity": average_velocity
        }

    def should_alert(self, alert_type: str, current_time: float,
                     min_interval: float = 5.0) -> bool:  # Aumentado intervalo mínimo
        """Determinar si se debe alertar según tipo y tiempo"""
        last_alert_time = self.last_alerted.get(alert_type, 0.0)
        return (current_time - last_alert_time) >= min_interval

    def mark_alerted(self, alert_type: str, timestamp: float):
        """Marcar que se ha alertado de este tipo"""
        self.last_alerted[alert_type] = timestamp

    def is_moving_significantly(self, threshold: float = 40.0) -> bool:
        """Verificar si el objeto se mueve significativamente"""
        return self.get_velocity() > threshold

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

class TemporalTracker:
    """Sistema avanzado de seguimiento temporal"""

    def __init__(self, max_objects: int = 100):
        self.tracked_objects: Dict[str, TemporalObject] = {}
        self.class_counters: Dict[str, int] = defaultdict(int)
        self.max_objects = max_objects
        self.detection_history = deque(maxlen=50)  # Historial de detecciones recientes
        self.frame_size = (640, 480)  # Tamaño por defecto del frame

    def _generate_track_id(self, class_name: str) -> str:
        """Generar ID único para seguimiento"""
        self.class_counters[class_name] += 1
        track_id = f"{class_name}_{self.class_counters[class_name]}_{int(time.time())}"
        return track_id

    def update_object(self, detection: Dict, frame_width: int, frame_height: int,
                      timestamp: float) -> 'TemporalObject':
        """Actualizar o crear objeto temporal"""
        class_name = detection['class']
        bbox = detection['bbox']
        distance = detection.get('distance_m', 5.0)
        risk_level = detection.get('risk_level', 'LOW')
        confidence = detection.get('confidence', 0.5)

        # Actualizar tamaño del frame
        self.frame_size = (frame_width, frame_height)

        # Verificar si es una detección duplicada reciente
        if self._is_duplicate_detection(detection, timestamp):
            # Encontrar el objeto existente más cercano
            track_id = self._match_to_existing(detection, frame_width, frame_height, timestamp)
            if track_id and track_id in self.tracked_objects:
                tracked_obj = self.tracked_objects[track_id]
                # Actualizar con nueva información si es más confiable
                if confidence > tracked_obj.confidence + 0.1:  # Solo si es significativamente más confiable
                    tracked_obj.update_observation(bbox, distance, risk_level,
                                                 frame_width, frame_height, timestamp, confidence)
                return tracked_obj
            else:
                # Crear nuevo objeto si no hay match
                pass
        else:
            # Registrar en historial
            self.detection_history.append({
                'class': class_name,
                'bbox': bbox,
                'timestamp': timestamp,
                'confidence': confidence
            })

        # Buscar objeto existente cercano
        track_id = self._match_to_existing(detection, frame_width, frame_height, timestamp)

        if track_id and track_id in self.tracked_objects:
            # Actualizar objeto existente
            tracked_obj = self.tracked_objects[track_id]
            tracked_obj.update_observation(bbox, distance, risk_level,
                                         frame_width, frame_height, timestamp, confidence)
        else:
            # Crear nuevo objeto
            track_id = self._generate_track_id(class_name)
            tracked_obj = TemporalObject(
                track_id=track_id,
                class_name=class_name,
                first_seen=timestamp,
                last_seen=timestamp,
                confidence=confidence
            )
            tracked_obj.update_observation(bbox, distance, risk_level,
                                         frame_width, frame_height, timestamp, confidence)
            self.tracked_objects[track_id] = tracked_obj

            # Limpiar objetos antiguos si se excede el límite
            if len(self.tracked_objects) > self.max_objects:
                self._cleanup_old_objects(timestamp)

        return tracked_obj

    def _is_duplicate_detection(self, detection: Dict, timestamp: float,
                                time_window: float = 1.5) -> bool:
        """Verificar si la detección es un duplicado reciente"""
        current_class = detection['class']
        current_bbox = detection['bbox']
        current_confidence = detection.get('confidence', 0.5)

        for recent_detection in reversed(self.detection_history):
            # Solo considerar detecciones recientes
            if timestamp - recent_detection['timestamp'] > time_window:
                continue

            # Mismo tipo de objeto
            if recent_detection['class'] != current_class:
                continue

            # Verificar si los bounding boxes se superponen significativamente
            if self._bboxes_overlap_significantly(current_bbox, recent_detection['bbox']):
                # Si la nueva detección no es significativamente mejor, considerarla duplicada
                if current_confidence <= recent_detection['confidence'] + 0.15:  # Aumentado umbral
                    return True

        return False

    def _bboxes_overlap_significantly(self, bbox1: List[int], bbox2: List[int],
                                      threshold: float = 0.65) -> bool:  # Aumentado umbral
        """Verificar si dos bounding boxes se superponen significativamente"""
        # Calcular IoU
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersección
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Áreas
        area1 = max(1, (x2_1 - x1_1) * (y2_1 - y1_1))
        area2 = max(1, (x2_2 - x1_2) * (y2_2 - y1_2))

        if area1 == 0 or area2 == 0:
            return False

        # IoU
        union_area = area1 + area2 - inter_area
        if union_area == 0:
            return False

        iou = inter_area / union_area
        return iou > threshold

    def _match_to_existing(self, detection: Dict, frame_width: int, frame_height: int,
                           timestamp: float, max_time_diff: float = 2.5) -> Optional[str]:
        """Encontrar objeto existente que coincida con la detección"""
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

            # Verificar si está activo (visto recientemente)
            if timestamp - tracked_obj.last_seen > max_time_diff:
                continue

            # Calcular distancia al último punto
            if tracked_obj.positions:
                last_pos = tracked_obj.positions[-1]
                distance = np.sqrt((current_cx - last_pos[0]) ** 2 + (current_cy - last_pos[1]) ** 2)

                # Normalizar por tamaño del frame
                frame_diagonal = np.sqrt(frame_width ** 2 + frame_height ** 2)
                normalized_distance = distance / frame_diagonal

                # Puntaje inverso a la distancia (más cerca = mejor coincidencia)
                # Aumentar el umbral de coincidencia
                score = 1.0 - min(1.0, normalized_distance)

                if score > best_score and score > 0.6:  # Umbral de coincidencia más permisivo
                    best_score = score
                    best_match_id = track_id

        return best_match_id

    def _cleanup_old_objects(self, current_time: float, max_age: float = 15.0):
        """Limpiar objetos antiguos"""
        expired_tracks = [
            track_id for track_id, obj in self.tracked_objects.items()
            if current_time - obj.last_seen > max_age
        ]

        for track_id in expired_tracks:
            del self.tracked_objects[track_id]

    def get_objects_in_zone(self, zone: str, frame_width: int,
                           frame_height: int) -> List[TemporalObject]:
        """Obtener objetos en una zona específica"""
        objects_in_zone = []

        for tracked_obj in self.tracked_objects.values():
            if tracked_obj.positions:
                last_pos = tracked_obj.positions[-1]
                cx, cy = last_pos[0], last_pos[1]

                # Determinar zona
                center_x = frame_width / 2
                if zone == "front":
                    if abs(cx - center_x) < frame_width * 0.3:  # Centro 60% del frame
                        objects_in_zone.append(tracked_obj)
                elif zone == "left":
                    if cx < center_x - frame_width * 0.15:
                        objects_in_zone.append(tracked_obj)
                elif zone == "right":
                    if cx > center_x + frame_width * 0.15:
                        objects_in_zone.append(tracked_obj)

        return objects_in_zone

    def get_objects_by_risk(self, risk_level: str) -> List[TemporalObject]:
        """Obtener objetos por nivel de riesgo"""
        objects_by_risk = []

        for tracked_obj in self.tracked_objects.values():
            if tracked_obj.risk_levels and tracked_obj.risk_levels[-1] == risk_level:
                objects_by_risk.append(tracked_obj)

        return objects_by_risk
