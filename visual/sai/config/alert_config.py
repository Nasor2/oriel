# visual/sai/config/alert_config.py
"""
AlertSystem Configuration
Contiene todos los thresholds, cooldowns, pesos de riesgo y parámetros de operación.
"""

# ============================================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================================

YOLO_CONFIG = {
    "model_path": "models/yolov8n.pt",  # YOLOv8n oficial (COCO dataset, 80 clases)
    "device": "cpu",  # Siempre CPU
    "conf_threshold": 0.45,  # Threshold de confianza YOLO
    "iou_threshold": 0.45,  # Threshold NMS
    "img_size": 640,  # Tamaño de entrada YOLO
    "batch_size": 1,  # Inference tiempo real
}

MIDAS_CONFIG = {
    "model_type": "DPT_Hybrid",  # Modelo MiDaS a usar
    "device": "cpu",  # Siempre CPU
    "img_size": 384,  # Tamaño de entrada MiDaS
}

# ============================================================================
# CLASES DETECTABLES (Solo clases relevantes en la calle - COCO)
# ============================================================================

DETECTABLE_CLASSES = [
    'person',        # Personas
    'bicycle',       # Bicicletas
    'car',          # Autos
    'motorcycle',    # Motos
    'airplane',      # Aviones
    'bus',          # Buses
    'train',        # Trenes
    'truck',        # Camiones
    'boat',         # Botes
    'dog',          # Perros
    'cat',          # Gatos
    'horse',        # Caballos
    'cow',          # Vacas
    'elephant',     # Elefantes
    'bear',         # Osos
]

# ============================================================================
# CLASIFICACIÓN DE RIESGO POR CLASE
# ============================================================================

CLASS_RISK_BASE = {
    # VEHÍCULOS GRANDES (ALTO RIESGO)
    'bus': 0.85,
    'truck': 0.85,
    'train': 0.80,
    'airplane': 0.75,
    'boat': 0.60,

    # MOTOCICLETAS Y BICICLETAS (RIESGO MEDIO)
    'motorcycle': 0.75,
    'bicycle': 0.50,

    # VEHÍCULOS PERSONALES (RIESGO MEDIO)
    'car': 0.65,

    # PERSONAS (RIESGO BAJO)
    'person': 0.30,

    # ANIMALES (RIESGO BAJO-MEDIO)
    'dog': 0.25,
    'cat': 0.20,
    'horse': 0.35,
    'cow': 0.40,
    'elephant': 0.50,
    'bear': 0.60,
}

# ============================================================================
# FACTORES DE RIESGO (PESOS)
# ============================================================================

RISK_WEIGHTS = {
    "class_risk": 0.10,      # 15% del score es la clase
    "proximity": 0.45,       # 50% es la distancia (aumentado de 45%)
    "velocity": 0.20,        # 25% es la velocidad de acercamiento (reducido de 30%)
    "trajectory": 0.25,      # 10% es la trayectoria
}

# Nota: suma = 1.0

# ============================================================================
# THRESHOLDS DE RIESGO (REDUCIDOS PARA ALERTAR MÁS PRONTO)
# ============================================================================

RISK_THRESHOLDS = {
    "LOW": (0.0, 0.30),      # risk_score entre 0.0-0.30 (ajustado de 0.25)
    "MEDIUM": (0.30, 0.65),  # risk_score entre 0.30-0.65 (ajustado de 0.55)
    "HIGH": (0.65, 1.0),     # risk_score entre 0.65-1.0 (ajustado de 0.55)
}

# ============================================================================
# COOLDOWNS Y DEDUPLICACIÓN
# ============================================================================

ALERT_COOLDOWNS = {
    "LOW": 20.0,      # Alertar cada 20s para bajo riesgo (aumentado de 15s)
    "MEDIUM": 15.0,   # Alertar cada 15s para riesgo medio (aumentado de 10s)
    "HIGH": 8.0,      # Alertar cada 8s para alto riesgo (aumentado de 5s)
}


DEDUP_WINDOW = 2.0  # Ventana temporal para considerar "mismo objeto" (segundos)

# ============================================================================
# CLUSTERING / AGRUPAMIENTO
# ============================================================================

CLUSTERING_CONFIG = {
    "algorithm": "distance",  # "distance" o "dbscan"
    "epsilon_m": 1.5,         # Distancia máxima para agrupar (metros)
    "min_samples": 1,         # Mínimo de objetos para formar grupo
}

# ============================================================================
# BUFFER CIRCULAR
# ============================================================================

FRAME_BUFFER_SIZE = 60  # Últimos 60 frames (@ 15 FPS = ~4 segundos)

# ============================================================================
# UMBRALES ESPACIALES
# ============================================================================

PROXIMITY_THRESHOLDS = {
    "very_close": 0.8,      # < 0.8m: muy cerca (reducido de 1.0m)
    "close": 2.0,           # 0.8-2.0m: cerca (reducido de 2.5m)
    "medium": 3.5,          # 2.0-3.5m: media distancia (reducido de 4.0m)
    "far": float('inf'),    # > 3.5m: lejos
}

# ============================================================================
# VELOCIDAD (píxeles por frame -> estimación)
# ============================================================================

VELOCITY_THRESHOLDS = {
    "stationary": 0.5,       # < 0.5 m/s: estacionario
    "slow": 2.0,             # 0.5-2.0 m/s: lento
    "medium": 5.0,           # 2.0-5.0 m/s: medio
    "fast": float('inf'),    # > 5.0 m/s: rápido
}

# ============================================================================
# CONTEXTO ESPACIAL (ÁNGULOS RELATIVOS A CÁMARA)
# ============================================================================

SPATIAL_CONTEXT = {
    "front": (0, 30),          # 0-30°: frente
    "lateral_left": (30, 90),  # 30-90°: a la izquierda
    "lateral_right": (-90, -30),  # -90 a -30°: a la derecha
    "back": (-180, -90),       # -180 a -90°: detrás
}

# ============================================================================
# GENERACIÓN DE MENSAJES (PLACEHOLDERS)
# ============================================================================

MESSAGE_TEMPLATES = {
    "LOW": {
        "single": "Ten cuidado con un {class_name}",
        "grouped": "Ten cuidado, hay {count} {class_plural} por aquí",
        "approaching": "Ten cuidado, {class_name} acercándose lentamente",
    },
    "MEDIUM": {
        "single": "Cuidado, hay un {class_name} {direction}",
        "grouped": "Cuidado, {count} {class_plural} alrededor",
        "approaching": "Cuidado, {class_name} acercándose desde {direction}",
        "high_density": "Cuidado, muchas {class_plural} alrededor, camina despacio",
    },
    "HIGH": {
        "single": "¡PELIGRO! {class_name} muy cerca",
        "grouped": "¡PELIGRO! Múltiples {class_plural} aproximándose",
        "approaching": "¡PELIGRO! {class_name} acercándose rápidamente desde {direction}",
        "imminent": "¡PELIGRO! Colisión inminente, detente",
    },
}

# ============================================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ============================================================================

ALERT_STREAM_FPS = 15  # FPS del procesamiento de alertas (balance velocidad/latencia)

INFERENCE_TIMEOUT = 5.0  # Timeout máximo para inferencia (segundos)

# ============================================================================
# MODOS DE OPERACIÓN
# ============================================================================

OPERATION_MODES = {
    "debug": {
        "verbose": True,
        "log_detections": True,
        "cooldowns": {k: v * 0.1 for k, v in ALERT_COOLDOWNS.items()},  # Sin cooldown
        "skip_dedup": False,
    },
    "production": {
        "verbose": False,
        "log_detections": False,
        "cooldowns": ALERT_COOLDOWNS,
        "skip_dedup": False,
    },
    "battery_saver": {
        "verbose": False,
        "log_detections": False,
        "cooldowns": {k: v * 2.0 for k, v in ALERT_COOLDOWNS.items()},  # Cooldowns 2x
        "alert_stream_fps": 10,  # Reducir FPS
        "skip_dedup": False,
    },
}

DEFAULT_MODE = "production"

