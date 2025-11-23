# core/contracts/visual_alert_event.py
from dataclasses import dataclass, field
import time
from typing import Optional

@dataclass
class VisualAlertEvent:
    """
    Payload que publica el AlertSubsystem hacia el Orchestrator.
    - frame_id: id del frame que generó la alerta (opcional)
    - message: texto directo para TTS
    - level: 'low' | 'medium' | 'high'
    - object_class: clase detectada (ej: 'car', 'person', ...)
    - distance_m: estimación (metros) si está disponible
    - bbox: [x1,y1,x2,y2] formato en coordenadas del frame (opcional)
    - meta: dict adicional (fps, track_id, velocity, etc)
    - timestamp: epoch
    """
    frame_id: Optional[str]
    message: str
    level: str = "medium"
    object_class: Optional[str] = None
    distance_m: Optional[float] = None
    bbox: Optional[list] = None
    meta: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: time.time())
