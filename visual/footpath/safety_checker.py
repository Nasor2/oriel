# visual/footpath/safety_checker.py
"""
Evaluador de seguridad para sugerencias de caminería.
"""

from colorama import Fore
from .config import SAFETY_THRESHOLDS

class SafetyChecker:
    """
    Evaluador de condiciones de seguridad para caminería.
    """

    def __init__(self):
        print(Fore.CYAN + "[SafetyChecker] Evaluador de seguridad inicializado.")

    def is_path_safe(self, analysis_result):
        """
        Verifica si el camino es seguro para sugerir avanzar.
        Relajado para coincidir mejor con el enfoque de inspiración.
        """
        # Verificar condiciones críticas de seguridad pero con umbrales más realistas
        confidence = analysis_result['confidence']
        continuity = analysis_result['path_continuity_ratio']
        n_components = analysis_result['n_components']
        has_large_jumps = analysis_result['has_large_jumps']
        mask_clipped = analysis_result['mask_clipped']
        distance_front = analysis_result['distance_front_ft']

        # Considerar seguro si:
        # 1. Confianza y continuidad mínima básicas se cumplen
        # 2. No hay múltiples componentes desconectados
        # 3. No hay muchos saltos grandes
        # 4. Hay algo de distancia frontal

        basic_conditions = [
            confidence > SAFETY_THRESHOLDS["min_confidence"],  # 0.25
            continuity > SAFETY_THRESHOLDS["min_continuity"],  # 0.15
            n_components <= SAFETY_THRESHOLDS["max_components"],  # 2
            not (has_large_jumps and n_components > 1),  # Permitir saltos si solo hay 1 componente
            distance_front > 0.1  # Algo de distancia
        ]

        # Si la máscara está cortada pero hay buena confianza y continuidad, aún puede ser usable
        if mask_clipped:
            # Permitir máscaras cortadas si otros indicadores son buenos
            is_safe = all(basic_conditions) and (confidence > 0.4 or continuity > 0.3)
        else:
            is_safe = all(basic_conditions)

        if not is_safe:
            print(Fore.YELLOW + f"[SafetyChecker] Camino no seguro: conf={confidence:.2f}, cont={continuity:.2f}")

        return is_safe

    def get_safety_violations(self, analysis_result):
        """
        Obtiene lista de violaciones de seguridad.

        Args:
            analysis_result: dict con resultados del PathAnalyzer

        Returns:
            list: Lista de violaciones encontradas
        """
        violations = []

        if analysis_result['path_continuity_ratio'] <= SAFETY_THRESHOLDS["min_continuity"]:
            violations.append("Baja continuidad del camino")

        if analysis_result['confidence'] <= SAFETY_THRESHOLDS["min_confidence"]:
            violations.append("Baja confianza en la detección")

        if analysis_result['n_components'] > SAFETY_THRESHOLDS["max_components"]:
            violations.append("Múltiples componentes de camino")

        if analysis_result['has_large_jumps']:
            violations.append("Saltos grandes en la trayectoria")

        if analysis_result['mask_clipped']:
            violations.append("Máscara cortada en bordes")

        if analysis_result['distance_front_ft'] <= SAFETY_THRESHOLDS["min_front_distance_ft"]:
            violations.append("Distancia frontal insuficiente")

        return violations
