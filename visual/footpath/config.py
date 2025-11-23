# visual/footpath/config.py
"""
Configuración del subsistema Footpath.
"""

# Parámetros del modelo
MODEL_CONFIG = {
    "input_size": (256, 256),
    "model_path": "models/qpulm_pruned_final.keras",  # Ruta al modelo .keras
}

# Parámetros de análisis SODD-CA
SODD_CA_CONFIG = {
    "bands": 8,
    "min_area_px": 60,
    "gap_tolerance_rows": 1,
    "centroid_vertical_gap_factor": 1.5,
    "continuity_threshold": 0.35,
    "final_conf_threshold_uncertain": 0.45,
    "final_conf_threshold_no_path": 0.25,
}

# Umbrales de seguridad
SAFETY_THRESHOLDS = {
    "min_confidence": 0.25,
    "min_continuity": 0.15,
    "max_components": 2,
    "max_jumps": 6,
    "min_front_distance_ft": 0.25,
}

# Factores de conversión (ft por píxel)
CONVERSION_FACTORS = {
    "alpha": 0.023,  # ft por píxel (ancho)
    "beta": 0.11,    # ft por píxel (alto)
}
