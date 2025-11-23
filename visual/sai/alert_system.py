# visual/sai/alert_system.py
import time
import os
from typing import Optional, List, Dict
from colorama import Fore

from core.contracts.visual_alert_event import VisualAlertEvent
from .pipeline.object_tracker import ObjectTracker, TrackedObject
from .pipeline.temporal_tracker import TemporalObject, TemporalTracker
from .pipeline.alert_manager import AlertManager, ContextualAlert
from .config import (
    YOLO_CONFIG,
    MIDAS_CONFIG,
    DETECTABLE_CLASSES,
    ALERT_STREAM_FPS,
    OPERATION_MODES,
    DEFAULT_MODE,
    CLASS_RISK_BASE,
    RISK_WEIGHTS,
    RISK_THRESHOLDS
)
from .engine import InferenceEngine
from .utils import (
    pixel_distance_to_meters,
    proximity_risk_score,
    velocity_risk_score,
    trajectory_risk_score,
    combined_risk_score,
    confidence_adjusted_risk
)

class AlertSystem:
    """
    Sistema de alertas mejorado con seguimiento avanzado y contextualización
    """

    def __init__(self, event_bus, video_manager=None,
                 mode: str = DEFAULT_MODE, verbose: bool = False,
                 lazy_load: bool = True):
        self.event_bus = event_bus
        self.video_manager = video_manager
        self.mode = mode
        self.verbose = verbose or OPERATION_MODES[mode].get("verbose", False)
        self.lazy_load = lazy_load

        # Estado
        self.enabled = False
        self.alert_stream_running = False

        # Componentes mejorados
        self._inference_engine = None
        self._object_tracker = ObjectTracker()
        self._temporal_tracker = TemporalTracker()
        self._alert_manager = AlertManager()

        # Estadísticas
        self._stats = {
            "frames_processed": 0,
            "detections_total": 0,
            "alerts_emitted": 0,
            "alerts_suppressed": 0,
            "inference_time_avg": 0.0,
        }

        print(Fore.CYAN + f"[AlertSystem] Inicializando en modo '{mode}'...")
        self._initialize()

    def _initialize(self):
        """Inicializar componentes"""
        try:
            # Cargar motor de inferencia
            if not self.lazy_load:
                self._load_inference_engine()

            print(Fore.GREEN + "[AlertSystem] Inicialización completada.")
        except Exception as e:
            print(Fore.RED + f"[AlertSystem] Error en inicialización: {e}")
            raise

    def _load_inference_engine(self):
        """Cargar motor de inferencia"""
        if self._inference_engine is not None:
            return

        try:
            model_path = YOLO_CONFIG["model_path"]
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.getcwd(), model_path)

            self._inference_engine = InferenceEngine(
                yolo_model_path=model_path,
                device=YOLO_CONFIG["device"],
                yolo_conf_threshold=YOLO_CONFIG["conf_threshold"],
                yolo_iou_threshold=YOLO_CONFIG["iou_threshold"],
                img_size=YOLO_CONFIG["img_size"],
                class_names=DETECTABLE_CLASSES,
            )
        except Exception as e:
            print(Fore.YELLOW + f"[AlertSystem] Fallo al cargar InferenceEngine: {e}")
            self._inference_engine = None

    def start(self):
        """Iniciar sistema de alertas"""
        if self.enabled:
            print(Fore.YELLOW + "[AlertSystem] Ya está activo.")
            return

        if self.video_manager is None:
            print(Fore.RED + "[AlertSystem] VideoManager no inyectado.")
            return

        try:
            self.enabled = True

            # Iniciar stream de alertas
            fps = OPERATION_MODES[self.mode].get("alert_stream_fps", ALERT_STREAM_FPS)
            try:
                self.video_manager.start_alert_stream(
                    callback_fn=self.process_frame,
                    fps=fps
                )
            except TypeError:
                self.video_manager.start_alert_stream(
                    callback=self.process_frame,
                    fps=fps
                )

            self.alert_stream_running = True
            print(Fore.GREEN + "[AlertSystem] Sistema de alertas ACTIVADO.")

        except Exception as e:
            print(Fore.RED + f"[AlertSystem] Error al iniciar: {e}")
            self.enabled = False

    def stop(self):
        """Detener sistema de alertas"""
        if not self.enabled:
            print(Fore.YELLOW + "[AlertSystem]  Ya está inactivo.")
            return

        try:
            self.enabled = False

            if self.video_manager and self.alert_stream_running:
                self.video_manager.stop_alert_stream()
                self.alert_stream_running = False

            # Limpiar estado
            self._object_tracker = ObjectTracker()
            self._temporal_tracker = TemporalTracker()
            self._alert_manager = AlertManager()

            print(Fore.GREEN + "[AlertSystem] Sistema de alertas DESACTIVADO.")

        except Exception as e:
            print(Fore.RED + f"[AlertSystem] Error al detener: {e}")

    def process_frame(self, frame_payload):
        """Procesar frame con sistema mejorado"""
        if not self.enabled:
            return

        # Cargar motor si es carga diferida
        if self._inference_engine is None:
            self._load_inference_engine()

        try:
            import time as time_module
            t_start = time_module.time()

            frame_id = frame_payload.frame_id
            original_frame = frame_payload.original_frame

            if original_frame is None:
                return

            # 1. INFERENCIA
            inference_result = self._inference_engine.infer(original_frame)
            detections = self._inference_engine.get_detections_dict(inference_result)

            if self.verbose:
                print(Fore.LIGHTBLACK_EX +
                      f"[AlertSystem] Frame {frame_id}: {len(detections)} detecciones")

            # 2. SEGUIMIENTO DE OBJETOS
            timestamp = time.time()
            tracked_detections = self._object_tracker.update(detections, timestamp)

            # 3. ANÁLISIS DE RIESGO Y GENERACIÓN DE ALERTAS
            frame_h, frame_w = original_frame.shape[:2]
            self._process_detections_for_alerts(tracked_detections, frame_w, frame_h, timestamp)

            # 4. ACTUALIZAR ESTADÍSTICAS
            t_end = time_module.time()
            elapsed_ms = (t_end - t_start) * 1000.0

            self._stats["frames_processed"] += 1
            self._stats["detections_total"] += len(detections)
            self._stats["inference_time_avg"] = (
                (self._stats["inference_time_avg"] + elapsed_ms) / 2.0
            )

        except Exception as e:
            print(Fore.RED + f"[AlertSystem] Error procesando frame: {e}")
            import traceback
            traceback.print_exc()

    def _process_detections_for_alerts(self, tracked_detections: List,
                                       frame_width: int, frame_height: int,
                                       timestamp: float):
        """Procesar detecciones para generar alertas inteligentes"""

        # 1. Actualizar seguimiento temporal y calcular riesgos
        temporal_objects = []
        for detection, track_id in tracked_detections:
            # Obtener datos básicos
            bbox = detection['bbox']
            depth_estimate = detection.get('depth_estimate', 0.5)
            # CORRECCIÓN: La llave correcta es 'class', no 'class_name'
            class_name = detection['class']

            # Calcular distancia métrica (usando la nueva geometría)
            bbox_width = bbox[2] - bbox[0]
            distance_m = pixel_distance_to_meters(
                bbox_width=bbox_width,
                depth_estimate=depth_estimate,
                class_name=class_name,
                focal_length=650.0
            )

            # Guardamos la distancia EN la detección AHORA para que _calculate_risk_level la use
            detection['distance_m'] = distance_m

            # Calcular nivel de riesgo (ahora usará la distancia ya calculada)
            risk_level = self._calculate_risk_level(detection, track_id, frame_width, frame_height)

            detection['risk_level'] = risk_level

            # Actualizar seguimiento temporal
            temporal_obj = self._temporal_tracker.update_object(
                detection, frame_width, frame_height, timestamp
            )
            temporal_objects.append(temporal_obj)

        # 2. Generar alerta contextualizada
        contextual_alert = self._alert_manager.generate_contextual_alert(
            temporal_objects, frame_width, frame_height, timestamp
        )

        # 3. Emitir alerta si es relevante
        if contextual_alert:
            self._emit_contextual_alert(contextual_alert)

    def _calculate_risk_level(self, detection: Dict, track_id: str,
                              frame_width: int, frame_height: int) -> str:
        """Calcular nivel de riesgo mejorado"""
        class_name = detection['class']
        confidence = detection['confidence']

        # CORRECCIÓN: Usar la distancia que ya calculamos en el paso anterior
        # Esto evita llamar a pixel_distance_to_meters dos veces y evita errores de argumentos
        distance_m = detection.get('distance_m', 5.0)

        # Factor 1: Riesgo base de clase
        class_risk = CLASS_RISK_BASE.get(class_name, 0.5)

        # Factor 2: Proximidad (Usando la distancia métrica real)
        proximity_risk = proximity_risk_score(distance_m, max_distance=8.0)

        # Factor 3: Movimiento (obtener del tracker temporal)
        tracked_obj = self._temporal_tracker.tracked_objects.get(track_id)
        velocity_risk = 0.0
        if tracked_obj and tracked_obj.is_moving_significantly(threshold=35.0):
            velocity = tracked_obj.get_velocity()
            velocity_risk = velocity_risk_score(velocity, max_velocity=120.0)

        # Factor 4: Trayectoria
        trajectory = "approaching" if tracked_obj and tracked_obj.is_moving_significantly(
            threshold=35.0) else "stationary"
        trajectory_risk = trajectory_risk_score(trajectory)

        # Combinar factores
        risk_score = combined_risk_score(
            class_risk=class_risk,
            proximity_risk=proximity_risk,
            velocity_risk=velocity_risk,
            trajectory_risk=trajectory_risk,
            weights=RISK_WEIGHTS
        )

        # Ajustar por confianza
        risk_score = confidence_adjusted_risk(risk_score, confidence)

        # Determinar nivel
        for level, (min_val, max_val) in RISK_THRESHOLDS.items():
            if min_val <= risk_score < max_val:
                return level

        return "HIGH" if risk_score >= RISK_THRESHOLDS["HIGH"][0] else "LOW"

    def _emit_contextual_alert(self, contextual_alert: ContextualAlert):
        """Emitir alerta contextualizada"""
        # Crear evento de alerta
        alert_event = VisualAlertEvent(
            frame_id=None,
            message=contextual_alert.message,
            level=contextual_alert.risk_level.lower(),
            object_class=None,
            distance_m=None,
            bbox=None,
            meta={
                'alert_type': contextual_alert.alert_type,
                'priority': contextual_alert.priority,
                'objects_involved': contextual_alert.objects_involved,
                'zone': contextual_alert.zone,
                'trend_info': contextual_alert.trend_info
            }
        )

        # Publicar alerta
        self.event_bus.publish("visual_alert", alert_event)
        self._stats["alerts_emitted"] += 1

        if self.verbose:
            color_map = {"high": Fore.LIGHTRED_EX, "medium": Fore.LIGHTYELLOW_EX, "low": Fore.LIGHTBLUE_EX}
            color = color_map.get(contextual_alert.risk_level.lower(), Fore.CYAN)
            print(f"{color}[ALERT] {contextual_alert.message} (Priority: {contextual_alert.priority})")

    def get_stats(self):
        """Obtener estadísticas del sistema"""
        return self._stats.copy()
