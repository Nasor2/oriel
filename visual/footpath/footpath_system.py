# visual/footpath/footpath_system.py
"""
Sistema principal del subsistema Footpath Request.
Coordina todos los componentes para análisis y recomendación de caminería.
"""

import time
from colorama import Fore
from core.contracts.visual_footpath_event import VisualFootpathEvent
from .inference_engine import FootpathInferenceEngine
from .path_analyzer import PathAnalyzer
from .safety_checker import SafetyChecker
from .navigation_advisor import NavigationAdvisor

class FootpathSystem:
    """
    Sistema principal para análisis y sugerencias de caminería.
    """

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.inference_engine = FootpathInferenceEngine()
        self.path_analyzer = PathAnalyzer()
        self.safety_checker = SafetyChecker()
        self.navigation_advisor = NavigationAdvisor()

        # Estado del sistema
        self.is_active = False
        self.pending_requests = []

        print(Fore.GREEN + "[FootpathSystem] Subsistema Footpath inicializado.")

    def activate(self):
        """
        Activa el subsistema de footpath.
        """
        self.is_active = True
        print(Fore.GREEN + "[FootpathSystem] Activado.")

    def deactivate(self):
        """
        Desactiva el subsistema de footpath.
        """
        self.is_active = False
        self.pending_requests.clear()
        print(Fore.YELLOW + "[FootpathSystem] Desactivado.")

    def process_footpath_request(self, video_manager, alert_system_enabled=False, visualize=False):
        """
        Procesa una solicitud de análisis de caminería.

        Args:
            video_manager: instancia de VideoManager
            alert_system_enabled: bool indicando si el sistema de alertas está activo
        """
        if not self.is_active:
            print(Fore.YELLOW + "[FootpathSystem] Sistema no activo.")
            return

        try:
            # Si el sistema de alertas está activo, almacenar solicitud para después
            if alert_system_enabled:
                print(Fore.LIGHTBLACK_EX + "[FootpathSystem] Sistema de alertas activo, almacenando solicitud.")
                self.pending_requests.append(time.time())
                return

            # Capturar frame
            payload = video_manager.capture_frame()
            if not payload:
                print(Fore.RED + "[FootpathSystem] No se pudo capturar frame.")
                return

            # Obtener vista procesada para footpath
            footpath_view = payload.get_view("footpath")
            if footpath_view is None:
                print(Fore.RED + "[FootpathSystem] Vista 'footpath' no disponible.")
                return

            # Realizar inferencia
            print(Fore.CYAN + "[FootpathSystem] Procesando solicitud de caminería...")
            mask = self.inference_engine.infer(footpath_view)

            # Analizar camino
            analysis_result = self.path_analyzer.analyze(mask)

            if visualize and hasattr(self.path_analyzer, 'visualize_analysis'):
                self.path_analyzer.visualize_analysis(mask, analysis_result)

            # Verificar seguridad
            is_safe = self.safety_checker.is_path_safe(analysis_result)

            # Generar recomendación
            recommendation = self.navigation_advisor.generate_recommendation(
                analysis_result, is_safe)

            # Crear evento
            footpath_event = VisualFootpathEvent(
                frame_id=payload.frame_id,
                message=recommendation["message"],
                risk_level=recommendation["risk_level"],
                action=recommendation["action"],
                direction=recommendation["direction"],
                metrics=analysis_result,
                timestamp=time.time()
            )

            # Publicar evento
            self.event_bus.publish("footpath_result", footpath_event)
            print(Fore.GREEN + f"[FootpathSystem] Recomendación generada: {recommendation['message']}")

        except Exception as e:
            print(Fore.RED + f"[FootpathSystem] Error procesando solicitud: {e}")
            # Publicar evento de error
            error_event = VisualFootpathEvent(
                frame_id=None,
                message="Error al analizar el camino. Por favor, intente nuevamente.",
                risk_level="high",
                action="stop",
                direction="error",
                metrics={},
                timestamp=time.time()
            )
            self.event_bus.publish("footpath_result", error_event)

    def process_pending_requests(self, video_manager):
        """
        Procesa solicitudes pendientes cuando el sistema de alertas se desactiva.

        Args:
            video_manager: instancia de VideoManager
        """
        if not self.pending_requests:
            return

        print(Fore.CYAN + f"[FootpathSystem] Procesando {len(self.pending_requests)} solicitudes pendientes...")

        # Process only the most recent request (others are likely redundant)
        if self.pending_requests:
            self.process_footpath_request(video_manager, alert_system_enabled=False)

        # Clear all pending requests
        self.pending_requests.clear()
