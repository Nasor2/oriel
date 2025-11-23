# visual/footpath/navigation_advisor.py
# visual/footpath/navigation_advisor.py
"""
Generador de recomendaciones de navegación basado en análisis de caminería.
"""

from colorama import Fore

class NavigationAdvisor:
    """
    Generador de recomendaciones de navegación segura.
    """

    def __init__(self):
        print(Fore.CYAN + "[NavigationAdvisor] Generador de recomendaciones inicializado.")

    def generate_recommendation(self, analysis_result, is_safe):
        """
        Genera una recomendación de navegación basada en el análisis.

        Args:
            analysis_result: dict con resultados del análisis
            is_safe: bool indicando si el camino es seguro

        Returns:
            dict: recomendación con mensaje, riesgo y acción
        """
        direction = analysis_result.get('direction_label', 'no_path')
        conf = analysis_result.get('confidence', 0.0)
        cont = analysis_result.get('path_continuity_ratio', 0.0)
        d_front = analysis_result.get('distance_front_ft', 0.0)

        # Determinar nivel de riesgo
        risk_level = self._determine_risk_level(conf, cont, is_safe)

        # Generar mensaje y acción
        if not is_safe or direction in ["no_path", "uncertain"]:
            message, action = self._generate_unsafe_recommendation(
                risk_level, analysis_result)
        else:
            message, action = self._generate_safe_recommendation(
                direction, d_front, conf, cont)

        return {
            "message": message,
            "risk_level": risk_level,
            "action": action,
            "direction": direction
        }

    def _determine_risk_level(self, confidence, continuity, is_safe):
        """
        Determina el nivel de riesgo basado en métricas.
        """
        if not is_safe:
            # Si no es seguro pero hay indicadores decentes, puede ser moderado
            if confidence > 0.3 and continuity > 0.2:
                return "moderate"
            return "high"
        elif confidence < 0.4 or continuity < 0.3:
            return "moderate"
        else:
            return "low"

    def _generate_unsafe_recommendation(self, risk_level, analysis_result):
        """
        Genera recomendación para situaciones no seguras.
        Basado en navigation_recommendation_v2 del código de inspiración.
        """
        direction = analysis_result.get('direction_label', 'no_path')
        conf = analysis_result.get('confidence', 0.0)
        cont = analysis_result.get('path_continuity_ratio', 0.0)

        if direction == 'no_sidewalk':
            return "No detecto ninguna acera o camino seguro aquí.", "stop"

        # Para caminos inciertos pero con alguna información
        if risk_level in ["moderate", "high"] and direction not in ["no_path", "uncertain"]:
            if direction == "straight":
                if cont > 0.3:
                    rec = "El camino adelante parece recto pero tiene irregularidades. Avanza con cuidado."
                    action = "proceed_carefully"
                else:
                    rec = "Camino recto adelante. Puedes continuar."
                    action = "proceed"

            elif "curve_left" in direction:
                rec = "La acera se curva hacia la izquierda. Gira gradualmente."
                action = "turn_left"

            elif "curve_right" in direction:
                rec = "La acera se curva hacia la derecha. Gira gradualmente."
                action = "turn_right"

            elif "veer_left" in direction:
                rec = "El camino se desvía ligeramente hacia la izquierda. Mantente hacia el borde izquierdo."
                action = "veer_left"

            elif "veer_right" in direction:
                rec = "El camino se desvía ligeramente hacia la derecha. Mantente hacia el borde derecho."
                action = "veer_right"

            else:
                rec = "El camino es parcialmente visible. Espera o avanza lentamente."
                action = "wait"

        else:
            # Para caminos verdaderamente no detectados
            if conf < 0.2 or cont < 0.15:
                rec = "No se detecta un camino claro. Por favor, detente y reorienta tu dirección."
                action = "stop"

            elif 0.15 <= cont < 0.3:
                rec = "El camino es parcialmente visible. Espera o avanza lentamente."
                action = "wait"

            else:
                rec = "Camino incierto adelante. Reevalúa tu dirección."
                action = "wait"

        return rec, action

    def _generate_safe_recommendation(self, direction, distance_front, confidence, continuity):
        """
        Genera recomendación para situaciones seguras.
        """
        if direction == "straight":
            if continuity > 0.5:
                message = "Camino recto y despejado adelante. Puedes continuar."
                action = "proceed"
            else:
                message = "Camino recto adelante con algunas irregularidades. Continúa con cuidado."
                action = "proceed_carefully"

        elif "curve_left" in direction:
            message = "El camino se curva hacia la izquierda. Gira gradualmente."
            action = "turn_left"

        elif "curve_right" in direction:
            message = "El camino se curva hacia la derecha. Gira gradualmente."
            action = "turn_right"

        elif "veer_left" in direction:
            message = "El camino se desvía ligeramente hacia la izquierda. Mantente hacia el borde izquierdo."
            action = "veer_left"

        elif "veer_right" in direction:
            message = "El camino se desvía ligeramente hacia la derecha. Mantente hacia el borde derecho."
            action = "veer_right"

        else:
            message = "Camino detectado pero la dirección es incierta. Procede con precaución."
            action = "proceed_cautiously"

        return message, action
