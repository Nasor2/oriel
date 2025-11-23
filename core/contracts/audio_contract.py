from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from typing import Dict, Any


@dataclass
class AudioIntentPayload:
    """
    Representa el resultado estructurado del procesamiento de audio.
    Este contrato define el formato estándar de salida del AudioPipeline.
    """
    source: str
    timestamp: str
    trace_id: str
    text: str
    intent: str
    confidence: float
    meta: Dict[str, Any]

    @classmethod
    def create(
        cls,
        text: str,
        intent: str,
        confidence: float,
        meta: Dict[str, Any] = None,
        source: str = "audio_pipeline",
    ):
        """
        Crea un nuevo payload con campos autogenerados (timestamp, trace_id).
        """
        return cls(
            source=source,
            timestamp=datetime.utcnow().isoformat(),
            trace_id=str(uuid.uuid4()),
            text=text,
            intent=intent,
            confidence=confidence,
            meta=meta or {},
        )

    def to_dict(self):
        """Convierte el payload a diccionario (para logs o JSON)."""
        return asdict(self)

    def is_valid(self) -> bool:
        """
        Retorna True si el payload tiene texto e intención válida.
        """
        return bool(self.text and self.intent and self.confidence > 0.2)
