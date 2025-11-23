# visual/sai/__init__.py
"""
SAI (Sistema de Alertas Inteligentes) - Subsistema Visual
Detección de objetos + análisis de riesgo + generación de alertas contextualizadas.
"""

from .alert_system import AlertSystem
from .config import (
    YOLO_CONFIG,
    MIDAS_CONFIG,
    DETECTABLE_CLASSES,
    CLASS_RISK_BASE,
    RISK_WEIGHTS,
    ALERT_COOLDOWNS,
)

__all__ = [
    "AlertSystem",
    "YOLO_CONFIG",
    "MIDAS_CONFIG",
    "DETECTABLE_CLASSES",
    "CLASS_RISK_BASE",
    "RISK_WEIGHTS",
    "ALERT_COOLDOWNS",
]

