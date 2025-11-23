def get_default_video_profiles():
    """
    Retorna un diccionario con las resoluciones estándar para cada modelo activo.
    """
    return {
        "yolo": (640, 640),  # Object detection - mayor resolución para mejor detección
        "midas": (384, 384),  # Depth estimation - tamaño óptimo para MiDaS
        "footpath": (256, 256),  # Walkable area segmentation
        "scene_vqa": (448, 448),  # Visual question answering (futuro)
    }
