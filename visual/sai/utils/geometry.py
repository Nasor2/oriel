# visual/sai/utils/geometry.py
"""
Utilidades geométricas para análisis espacial.
Implementa el modelo de cámara estenopeica (Pinhole Camera Model)
para estimación métrica robusta.
"""

import numpy as np
from typing import Tuple, List


def bbox_to_centroid(bbox: List[int]) -> Tuple[float, float]:
    """
    Convierte bbox [x1, y1, x2, y2] a centroide (cx, cy).
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def bbox_area(bbox: List[int]) -> float:
    """
    Calcula el área de un bbox en píxeles cuadrados.
    """
    x1, y1, x2, y2 = bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width * height


def bbox_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calcula IoU (Intersection over Union) entre dos bboxes.
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersección
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Unión
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula distancia euclidiana entre dos puntos 2D en píxeles.
    """
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def pixel_distance_to_meters(bbox_width: float, depth_estimate: float,
                             class_name: str = "unknown",
                             focal_length: float = 650.0) -> float:
    """
    Estima la profundidad (distancia Z) en metros combinando
    geometría proyectiva (modelo Pinhole) y estimación MiDaS.

    Args:
        bbox_width: ancho del bounding box en píxeles (obligatorio para el cálculo matemático)
        depth_estimate: valor crudo de MiDaS (0.0 - 1.0)
        class_name: clase del objeto ('person', 'car', etc.)
        focal_length: longitud focal aproximada (default 650 para webcam estándar 640x480)

    Returns:
        Distancia estimada en metros.
    """

    # 1. Definición de anchos reales promedio (en metros)
    # Estos valores actúan como anclas de realidad
    REAL_WIDTHS = {
        "person": 0.5,  # Ancho de hombros promedio
        "bicycle": 0.6,
        "motorcycle": 0.75,
        "car": 1.8,
        "taxi": 1.8,
        "bus": 2.55,
        "truck": 2.5,
        "train": 3.0,
        "dog": 0.3,  # Promedio entre razas
        "cat": 0.2,
        "horse": 0.7,
        "cow": 0.8,
        "bear": 0.8,
        "elephant": 1.5,
        "bench": 1.5,
        "chair": 0.5,
        "unknown": 1.0
    }

    # Distancia mínima y máxima lógica
    MIN_DIST = 0.3
    MAX_DIST = 25.0

    # 2. Cálculo Matemático (Modelo de Cámara Estenopeica)
    # Distancia = (Ancho_Real * Focal) / Ancho_Píxeles
    math_dist = 0.0
    has_math_dist = False

    real_width = REAL_WIDTHS.get(class_name, REAL_WIDTHS["unknown"])

    if bbox_width > 10:  # Evitar división por cero o ruido muy pequeño
        math_dist = (real_width * focal_length) / bbox_width
        has_math_dist = True

    # 3. Cálculo basado en MiDaS (Heurístico)
    # Invertimos y escalamos arbitrariamente para cuando falla la matemática
    inverted_depth = 1.0 - depth_estimate
    # Ecuación empírica ajustada
    midas_dist = MIN_DIST + (MAX_DIST - MIN_DIST) * (inverted_depth ** 1.5)

    # 4. Fusión de Sensores (Lógica Híbrida)
    final_distance = midas_dist  # Default

    if has_math_dist and class_name in REAL_WIDTHS:
        # Si conocemos el objeto, confiamos mayormente en la matemática
        # Peso 80% matemáticas, 20% MiDaS (para suavizar errores de bbox inestable)
        final_distance = (math_dist * 0.80) + (midas_dist * 0.20)

    elif has_math_dist:
        # Objeto desconocido pero tenemos bbox válido
        # Peso 50/50
        final_distance = (math_dist * 0.50) + (midas_dist * 0.50)

    return np.clip(final_distance, MIN_DIST, MAX_DIST)


def centroid_distance_meters(bbox1: List[int], bbox2: List[int],
                             depth_m_1: float, depth_m_2: float,
                             focal_length: float = 650.0) -> float:
    """
    Calcula la distancia real 3D entre dos objetos en metros.
    Utiliza la profundidad métrica ya calculada.

    Args:
        bbox1, bbox2: bounding boxes
        depth_m_1, depth_m_2: distancias en metros (YA calculadas con pixel_distance_to_meters)
        focal_length: longitud focal

    Returns:
        Distancia euclidiana 3D en metros
    """
    c1 = bbox_to_centroid(bbox1)
    c2 = bbox_to_centroid(bbox2)

    # 1. Distancia lateral en píxeles (Plano X-Y)
    pixel_dist_lateral = euclidean_distance(c1, c2)

    # 2. Convertir distancia lateral a metros usando la profundidad promedio
    # Teorema de tales: Dist_m = (Dist_px * Profundidad_m) / Focal
    avg_depth = (depth_m_1 + depth_m_2) / 2.0
    lateral_dist_m = (pixel_dist_lateral * avg_depth) / focal_length

    # 3. Distancia longitudinal (Plano Z)
    longitudinal_dist_m = abs(depth_m_1 - depth_m_2)

    # 4. Distancia Euclidian 3D total (Hipotenusa)
    return np.sqrt(lateral_dist_m ** 2 + longitudinal_dist_m ** 2)


def angle_from_center(bbox: List[int], frame_width: int, frame_height: int) -> float:
    """
    Calcula ángulo relativo del objeto respecto al centro de la cámara.
    Args:
        bbox: [x1, y1, x2, y2]
        frame_width: ancho del frame
    Returns:
        ángulo en grados (-180 a 180)
    """
    cx, _ = bbox_to_centroid(bbox)
    center_x = frame_width / 2.0

    # Calcular desviación del centro
    dx = cx - center_x

    # Aproximación lineal de ángulo basada en FOV horizontal (~60 grados para webcams)
    # FOV_H / 2 = 30 grados
    fov_half = 30.0

    # Regla de tres: (dx / (width/2)) * 30 grados
    angle = (dx / center_x) * fov_half

    return np.clip(angle, -90, 90)


def direction_label(angle: float) -> str:
    """
    Convierte ángulo a etiqueta de dirección textual.
    """
    if -20 <= angle <= 20:  # Reduje el ángulo central para ser más preciso
        return "frente"
    elif 20 < angle <= 60:
        return "derecha"
    elif angle > 60:
        return "muy a la derecha"
    elif -60 <= angle < -20:
        return "izquierda"
    else:
        return "muy a la izquierda"


def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> bool:
    """
    Verifica si dos bboxes se solapan significativamente.
    """
    return bbox_iou(bbox1, bbox2) > 0.1