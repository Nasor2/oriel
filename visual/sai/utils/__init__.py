# visual/sai/utils/__init__.py
"""
Paquete de utilidades para el sistema SAI.
"""

from .geometry import (
    bbox_to_centroid,
    bbox_area,
    bbox_iou,
    euclidean_distance,
    pixel_distance_to_meters,
    centroid_distance_meters,
    angle_from_center,
    direction_label,
    bbox_overlap
)

from .metrics import (
    estimate_velocity_px_per_frame,
    estimate_trajectory,
    proximity_risk_score,
    velocity_risk_score,
    trajectory_risk_score,
    combined_risk_score,
    confidence_adjusted_risk,
    calculate_frame_density,
    is_significant_movement,
    calculate_speed_in_mps
)

__all__ = [
    'bbox_to_centroid',
    'bbox_area',
    'bbox_iou',
    'euclidean_distance',
    'pixel_distance_to_meters',
    'centroid_distance_meters',
    'angle_from_center',
    'direction_label',
    'bbox_overlap',
    'estimate_velocity_px_per_frame',
    'estimate_trajectory',
    'proximity_risk_score',
    'velocity_risk_score',
    'trajectory_risk_score',
    'combined_risk_score',
    'confidence_adjusted_risk',
    'calculate_frame_density',
    'is_significant_movement',
    'calculate_speed_in_mps'
]
