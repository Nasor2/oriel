# core/contracts/visual_footpath_event.py
"""
Evento de contrato para resultados del subsistema Footpath.
"""

from dataclasses import dataclass, field
import time
from typing import Optional, Dict

@dataclass
class VisualFootpathEvent:
    """
    Payload que publica el FootpathSubsystem hacia el Orchestrator.

    Attributes:
        frame_id: ID del frame que generó el análisis (opcional)
        message: Texto directo para TTS
        risk_level: 'low' | 'moderate' | 'high'
        action: Acción recomendada ('proceed', 'stop', 'wait', etc.)
        direction: Dirección del camino ('straight', 'curve_left', etc.)
        metrics: Métricas detalladas del análisis
        timestamp: Epoch time
    """
    frame_id: Optional[str]
    message: str
    risk_level: str = "moderate"
    action: str = "wait"
    direction: str = "uncertain"
    metrics: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
