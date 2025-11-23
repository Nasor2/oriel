# visual/sai/utils/metrics.py
"""
Cálculos de métricas: velocidad, trayectoria, riesgo.
"""

import numpy as np
from typing import Tuple, List, Optional
from .geometry import bbox_to_centroid, euclidean_distance


def estimate_velocity_px_per_frame(current_pos: Tuple[float, float],
                                   previous_pos: Tuple[float, float]) -> float:
    """
    Estima velocidad en píxeles/frame.

    Args:
        current_pos: (cx, cy) posición actual
        previous_pos: (cx, cy) posición anterior

    Returns:
        velocidad en píxeles/frame
    """
    if previous_pos is None:
        return 0.0

    dist = euclidean_distance(current_pos, previous_pos)
    return max(0.0, dist)


def estimate_trajectory(current_pos: Tuple[float, float],
                       previous_pos: Tuple[float, float],
                       frame_center: Tuple[float, float]) -> str:
    """
    Clasifica trayectoria: acercándose, alejándose, lateral.

    Args:
        current_pos: (cx, cy) posición actual
        previous_pos: (cx, cy) posición anterior
        frame_center: (cx_center, cy_center) centro del frame

    Returns:
        "approaching", "receding", "lateral"
    """
    if previous_pos is None:
        return "unknown"

    # Distancia anterior y actual al centro
    dist_prev = euclidean_distance(previous_pos, frame_center)
    dist_curr = euclidean_distance(current_pos, frame_center)

    # Diferencia de distancia
    delta = dist_curr - dist_prev

    if abs(delta) < 5:  # Umbral de ruido
        return "lateral"
    elif delta < 0:
        return "approaching"
    else:
        return "receding"


def proximity_risk_score(distance_m: float, max_distance: float = 10.0) -> float:
    """
    Calcula score de riesgo por proximidad (0.0-1.0).

    Args:
        distance_m: distancia en metros
        max_distance: distancia máxima considerada (default 10m)

    Returns:
        risk_score (0.0-1.0)
    """
    if distance_m < 0:
        return 0.0

    # Función inversa: más cerca = más riesgo
    # f(d) = 1 / (1 + (d/max_d)^2)
    normalized_distance = distance_m / max_distance
    risk = 1.0 / (1.0 + normalized_distance**2)
    return np.clip(risk, 0.0, 1.0)

def velocity_risk_score(velocity_px_per_frame: float, max_velocity: float = 50.0) -> float:
    """
    Calcula score de riesgo por velocidad.

    Args:
        velocity_px_per_frame: velocidad en píxeles/frame
        max_velocity: velocidad máxima considerada (default 50 px/frame)

    Returns:
        risk_score (0.0-1.0)
    """
    if velocity_px_per_frame < 0:
        return 0.0

    # Función cuadrática: f(v) = (v / max_velocity)^2
    normalized_velocity = velocity_px_per_frame / max_velocity
    risk = normalized_velocity**2
    return np.clip(risk, 0.0, 1.0)

def trajectory_risk_score(trajectory: str) -> float:
    """
    Calcula score de riesgo por trayectoria.

    Args:
        trajectory: "approaching", "receding", "lateral", "unknown"

    Returns:
        risk_score (0.0-1.0)
    """
    risk_map = {
        "approaching": 0.9,  # Aumentado riesgo para objetos que se acercan
        "lateral": 0.3,
        "receding": 0.1,
        "unknown": 0.2,
    }
    return risk_map.get(trajectory, 0.2)

def combined_risk_score(class_risk: float,
                       proximity_risk: float,
                       velocity_risk: float,
                       trajectory_risk: float,
                       weights: dict) -> float:
    """
    Calcula score de riesgo combinado ponderado.

    Args:
        class_risk: riesgo base de la clase (0.0-1.0)
        proximity_risk: riesgo por proximidad (0.0-1.0)
        velocity_risk: riesgo por velocidad (0.0-1.0)
        trajectory_risk: riesgo por trayectoria (0.0-1.0)
        weights: dict con pesos (deben sumar 1.0)

    Returns:
        risk_score combinado (0.0-1.0)
    """
    score = (
        weights.get("class_risk", 0.15) * class_risk +
        weights.get("proximity", 0.45) * proximity_risk +
        weights.get("velocity", 0.30) * velocity_risk +
        weights.get("trajectory", 0.10) * trajectory_risk
    )
    return np.clip(score, 0.0, 1.0)


def confidence_adjusted_risk(base_risk: float, confidence: float) -> float:
    """
    Ajusta score de riesgo por confianza de detección.

    Args:
        base_risk: score de riesgo base
        confidence: confianza YOLO (0.0-1.0)

    Returns:
        risk_score ajustado
    """
    # Si confianza baja, reducir riesgo
    return base_risk * np.clip(confidence, 0.4, 1.0)


def calculate_frame_density(detections: List[dict],
                           frame_width: int,
                           frame_height: int) -> float:
    """
    Calcula densidad de detecciones en el frame (0.0-1.0).

    Args:
        detections: lista de detecciones con 'bbox'
        frame_width: ancho del frame
        frame_height: alto del frame

    Returns:
        densidad (0.0-1.0)
    """
    if not detections:
        return 0.0

    frame_area = frame_width * frame_height
    total_detection_area = 0.0

    for det in detections:
        if 'bbox' in det:
            x1, y1, x2, y2 = det['bbox']
            area = max(0, x2 - x1) * max(0, y2 - y1)
            total_detection_area += area

    density = min(1.0, total_detection_area / frame_area)
    return density

def is_significant_movement(current_bbox: List[int],
                          previous_bbox: List[int],
                          frame_width: int,
                          frame_height: int,
                          threshold_ratio: float = 0.02) -> bool:
    """
    Determina si el movimiento entre dos bounding boxes es significativo.

    Args:
        current_bbox: [x1, y1, x2, y2] bounding box actual
        previous_bbox: [x1, y1, x2, y2] bounding box anterior
        frame_width: ancho del frame
        frame_height: alto del frame
        threshold_ratio: ratio del tamaño del frame para considerar movimiento significativo

    Returns:
        bool: True si el movimiento es significativo
    """
    from .geometry import bbox_to_centroid, euclidean_distance

    current_center = bbox_to_centroid(current_bbox)
    previous_center = bbox_to_centroid(previous_bbox)

    distance = euclidean_distance(current_center, previous_center)

    # Calcular threshold basado en el tamaño del frame
    frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
    threshold = frame_diagonal * threshold_ratio

    return distance > threshold

def calculate_speed_in_mps(current_bbox: List[int],
                         previous_bbox: List[int],
                         time_delta: float,
                         current_depth: float,
                         previous_depth: float) -> float:
    """
    Calcula la velocidad real del objeto en metros por segundo.

    Args:
        current_bbox: [x1, y1, x2, y2] bounding box actual
        previous_bbox: [x1, y1, x2, y2] bounding box anterior
        time_delta: diferencia de tiempo entre frames en segundos
        current_depth: profundidad actual estimada
        previous_depth: profundidad anterior estimada

    Returns:
        float: velocidad en metros por segundo
    """
    from .geometry import bbox_to_centroid, euclidean_distance, pixel_distance_to_meters

    if time_delta <= 0:
        return 0.0

    current_center = bbox_to_centroid(current_bbox)
    previous_center = bbox_to_centroid(previous_bbox)

    # Calcular distancia en píxeles
    pixel_distance = euclidean_distance(current_center, previous_center)

    # Promediar profundidades para estimar distancia real
    avg_depth = (current_depth + previous_depth) / 2.0

    # Convertir a metros
    meters_distance = pixel_distance_to_meters(pixel_distance, avg_depth)

    # Calcular velocidad en m/s
    if time_delta > 0:
        speed_mps = meters_distance / time_delta
        return speed_mps
    else:
        return 0.0

