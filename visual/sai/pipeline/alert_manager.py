# visual/sai/pipeline/alert_manager.py
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import numpy as np

@dataclass
class ContextualAlert:
    """Alerta contextualizada con información completa"""
    alert_type: str  # "immediate_danger", "caution", "observation", "group_warning"
    message: str
    priority: int  # 1-10 (10 más alta)
    objects_involved: List[str]  # IDs de objetos
    zone: str  # "front", "left", "right", "surroundings"
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    trend_info: Dict  # Información de tendencias
    timestamp: float

class AlertManager:
    """Gestor avanzado de alertas contextuales"""

    def __init__(self):
        self.recent_alerts = deque(maxlen=50)
        self.zone_alert_history = defaultdict(lambda: deque(maxlen=10))  # Historial por zona
        self.alert_suppression = {}  # Supresión temporal de alertas
        self.object_clusters = {}  # Agrupamiento de objetos
        self.last_group_alert_time = 0  # Última alerta de grupo

    def generate_contextual_alert(self, temporal_objects: List,
                                 frame_width: int, frame_height: int,
                                 timestamp: float) -> Optional[ContextualAlert]:
        """Generar alerta contextualizada basada en objetos y contexto"""

        # 1. Agrupar objetos cercanos del mismo tipo
        clustered_objects = self._cluster_similar_objects(temporal_objects, frame_width, frame_height)

        # 2. Analizar objetos por zonas
        front_objects = self._get_objects_in_front(clustered_objects, frame_width, frame_height)
        left_objects = self._get_objects_in_left(clustered_objects, frame_width, frame_height)
        right_objects = self._get_objects_in_right(clustered_objects, frame_width, frame_height)

        # 3. Verificar alertas de grupo (prioridad más alta)
        group_alert = self._check_group_alerts(front_objects, timestamp)
        if group_alert:
            self._update_alert_history(group_alert, timestamp)
            return group_alert

        # 4. Verificar alertas de peligro inminente
        danger_alert = self._check_immediate_danger(front_objects, timestamp)
        if danger_alert:
            self._update_alert_history(danger_alert, timestamp)
            return danger_alert

        # 5. Verificar alertas de precaución
        caution_alert = self._check_caution_alerts(front_objects, left_objects, right_objects, timestamp)
        if caution_alert:
            self._update_alert_history(caution_alert, timestamp)
            return caution_alert

        # 6. Alertas de observación (solo si no hay alertas recientes)
        if not self._has_recent_high_priority_alerts(timestamp, 3.0):
            observation_alert = self._check_observation_alerts(front_objects, timestamp)
            if observation_alert:
                self._update_alert_history(observation_alert, timestamp)
                return observation_alert

        return None

    def _cluster_similar_objects(self, objects: List, frame_width: int, frame_height: int) -> List:
        """Agrupar objetos cercanos del mismo tipo"""
        if len(objects) < 2:
            return objects

        # Agrupar objetos por clase
        class_groups = defaultdict(list)
        for obj in objects:
            class_groups[obj.class_name].append(obj)

        clustered_objects = []
        cluster_id = 0

        for class_name, class_objects in class_groups.items():
            if len(class_objects) < 2:
                # No agrupar objetos únicos
                clustered_objects.extend(class_objects)
                continue

            # Agrupar objetos cercanos
            used_objects = set()
            frame_diagonal = np.sqrt(frame_width**2 + frame_height**2)
            proximity_threshold = frame_diagonal * 0.15  # 15% del tamaño del frame

            for i, obj1 in enumerate(class_objects):
                if i in used_objects:
                    continue

                cluster = [obj1]
                used_objects.add(i)

                # Buscar objetos cercanos
                for j, obj2 in enumerate(class_objects[i+1:], i+1):
                    if j in used_objects:
                        continue

                    if self._are_objects_close(obj1, obj2, proximity_threshold):
                        cluster.append(obj2)
                        used_objects.add(j)

                # Si hay más de un objeto en el cluster, crear cluster representativo
                if len(cluster) > 1:
                    representative = self._create_cluster_representative(cluster, cluster_id)
                    clustered_objects.append(representative)
                    cluster_id += 1
                else:
                    clustered_objects.extend(cluster)

        return clustered_objects

    def _are_objects_close(self, obj1, obj2, threshold: float) -> bool:
        """Verificar si dos objetos están cerca"""
        if not obj1.positions or not obj2.positions:
            return False

        pos1 = obj1.positions[-1][:2]  # (cx, cy)
        pos2 = obj2.positions[-1][:2]  # (cx, cy)

        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance < threshold

    def _create_cluster_representative(self, cluster: List, cluster_id: int):
        """Crear un objeto representativo para un cluster"""
        from .temporal_tracker import TemporalObject

        # Tomar el objeto con mayor confianza como representante
        representative = max(cluster, key=lambda obj: getattr(obj, 'confidence', 0.5))

        # Crear un nuevo objeto que represente al cluster
        cluster_obj = TemporalObject(
            track_id=f"cluster_{cluster_id}",
            class_name=representative.class_name,
            first_seen=min(obj.first_seen for obj in cluster),
            last_seen=max(obj.last_seen for obj in cluster)
        )

        # Copiar datos relevantes del representante
        cluster_obj.positions = representative.positions
        cluster_obj.velocities = representative.velocities
        cluster_obj.distances = representative.distances
        cluster_obj.risk_levels = representative.risk_levels
        cluster_obj.last_alerted = representative.last_alerted

        return cluster_obj

    def _has_recent_high_priority_alerts(self, timestamp: float, timeframe: float = 3.0) -> bool:
        """Verificar si hay alertas de alta prioridad recientes"""
        if not self.recent_alerts:
            return False

        for alert in reversed(self.recent_alerts):
            if alert.timestamp > (timestamp - timeframe) and alert.priority >= 7:
                return True
        return False

    def _update_alert_history(self, alert: ContextualAlert, timestamp: float):
        """Actualizar historial de alertas"""
        self.recent_alerts.append(alert)
        self.zone_alert_history[alert.zone].append((alert, timestamp))

    def _check_group_alerts(self, front_objects: List, timestamp: float) -> Optional[ContextualAlert]:
        """Verificar alertas de grupo de objetos"""
        # Verificar si han pasado suficiente tiempo desde la última alerta de grupo
        if timestamp - self.last_group_alert_time < 12.0:  # Aumentado tiempo entre alertas
            return None

        # Agrupar por clase
        class_groups = defaultdict(list)
        for obj in front_objects:
            class_groups[obj.class_name].append(obj)

        # Verificar grupos grandes (>3 objetos)
        for class_name, objects in class_groups.items():
            if len(objects) >= 5:  # Aumentado umbral para grupos
                spanish_name = self._get_spanish_name(class_name)
                plural_name = self._get_plural(spanish_name)

                # Calcular distancia promedio del grupo
                avg_distance = np.mean([obj.distances[-1] if obj.distances else 5.0 for obj in objects])

                # Solo alertar si el grupo está relativamente cerca
                if avg_distance < 5.0:  # Reducida distancia para alerta
                    message = f"Cuidado, hay un grupo de {len(objects)} {plural_name} al frente"
                    priority = 8 if avg_distance < 2.5 else 6  # Ajuste de prioridad

                    self.last_group_alert_time = timestamp

                    return ContextualAlert(
                        alert_type="group_warning",
                        message=message,
                        priority=priority,
                        objects_involved=[obj.track_id for obj in objects],
                        zone="front",
                        risk_level="MEDIUM" if avg_distance < 2.5 else "LOW",
                        trend_info={},
                        timestamp=timestamp
                    )

        return None

    def _check_immediate_danger(self, front_objects: List, timestamp: float) -> Optional[ContextualAlert]:
        """Verificar peligro inminente"""
        for obj in front_objects:

            if len(obj.positions) < 5:
                continue

            trend = obj.get_trend_analysis()

            # Objeto acercándose rápidamente y muy cerca
            avg_distance = trend.get('average_distance', 5.0)
            avg_velocity = trend.get('average_velocity', 0.0)

            if (trend.get('distance_trend') == 'approaching' and
                    avg_distance < 1.0 and  # Reducir distancia para peligro inminente
                    avg_velocity > 70):  # Aumentada velocidad para alerta

                spanish_name = self._get_spanish_name(obj.class_name)
                article = self._get_article(spanish_name)

                message = f"¡PELIGRO INMINENTE! {article.capitalize()} {spanish_name} se acerca rápidamente, ¡detente!"
                return ContextualAlert(
                    alert_type="immediate_danger",
                    message=message,
                    priority=10,
                    objects_involved=[obj.track_id],
                    zone="front",
                    risk_level="HIGH",
                    trend_info=trend,
                    timestamp=timestamp
                )

            # Objeto muy cerca (< 0.3m)
            if avg_distance < 0.3:
                spanish_name = self._get_spanish_name(obj.class_name)
                article = self._get_article(spanish_name)

                message = f"¡ALERTA! {article.capitalize()} {spanish_name} muy cerca, ¡cuidado!"
                return ContextualAlert(
                    alert_type="immediate_danger",
                    message=message,
                    priority=9,
                    objects_involved=[obj.track_id],
                    zone="front",
                    risk_level="HIGH",
                    trend_info=trend,
                    timestamp=timestamp
                )

        return None

    def _check_caution_alerts(self, front_objects: List, left_objects: List,
                              right_objects: List, timestamp: float) -> Optional[ContextualAlert]:
        """Verificar alertas de precaución"""
        # Verificar objetos en frente a distancia media
        for obj in front_objects:
            trend = obj.get_trend_analysis()
            avg_distance = trend.get('average_distance', 5.0)
            risk_trend = trend.get('risk_trend', 'stable')

            # Ajustar cooldown para alertas de precaución
            if (0.8 <= avg_distance <= 2.0 and  # Ajustado rango de distancia
                    obj.should_alert("caution", timestamp, min_interval=8.0)):  # Aumentado cooldown

                spanish_name = self._get_spanish_name(obj.class_name)
                article = self._get_article(spanish_name)

                # Verificar tendencia de riesgo
                if risk_trend == 'increasing':
                    message = f"Cuidado, {article} {spanish_name} se acerca"
                else:
                    message = f"Hay {article} {spanish_name} al frente a {int(avg_distance)} metros"

                obj.mark_alerted("caution", timestamp)

                return ContextualAlert(
                    alert_type="caution",
                    message=message,
                    priority=7 if risk_trend == 'increasing' else 5,
                    objects_involved=[obj.track_id],
                    zone="front",
                    risk_level="MEDIUM" if risk_trend == 'increasing' else "LOW",
                    trend_info=trend,
                    timestamp=timestamp
                )

        # Verificar objetos en laterales (solo para objetos de alto riesgo)
        lateral_objects = left_objects + right_objects
        if lateral_objects:
            high_risk_lateral = [obj for obj in lateral_objects
                                 if obj.risk_levels and obj.risk_levels[-1] in ["HIGH"]]

            if high_risk_lateral and len(high_risk_lateral) >= 3:  # Aumentado umbral
                message = "Atención, varios objetos peligrosos en los laterales"
                return ContextualAlert(
                    alert_type="caution",
                    message=message,
                    priority=7,
                    objects_involved=[obj.track_id for obj in high_risk_lateral],
                    zone="surroundings",
                    risk_level="MEDIUM",
                    trend_info={},
                    timestamp=timestamp
                )

        return None

    def _check_observation_alerts(self, front_objects: List, timestamp: float) -> Optional[ContextualAlert]:
        """Verificar alertas de observación (información útil)"""
        # Solo alertar objetos únicos e importantes
        important_objects = []
        for obj in front_objects:
            trend = obj.get_trend_analysis()
            avg_distance = trend.get('average_distance', 5.0)

            # Solo objetos importantes y a distancia segura
            if avg_distance > 3.0 and obj.class_name in ['person', 'car', 'bicycle', 'dog', 'bus', 'motorcycle']:
                important_objects.append((obj, avg_distance))

        # Ordenar por prioridad (más cercanos primero)
        important_objects.sort(key=lambda x: x[1])

        # Solo alertar el objeto más cercano e importante
        if important_objects:
            obj, distance = important_objects[0]

            # Verificar cooldown más estricto para observaciones
            if obj.should_alert("observation", timestamp, min_interval=20.0):  # Aumentado cooldown
                spanish_name = self._get_spanish_name(obj.class_name)
                article = self._get_article(spanish_name)

                message = f"Hay {article} {spanish_name} al frente a {int(distance)} metros"
                obj.mark_alerted("observation", timestamp)

                return ContextualAlert(
                    alert_type="observation",
                    message=message,
                    priority=3,
                    objects_involved=[obj.track_id],
                    zone="front",
                    risk_level="LOW",
                    trend_info={},
                    timestamp=timestamp
                )

        return None


    def _get_objects_in_front(self, objects: List, frame_width: int, frame_height: int) -> List:
        """Obtener objetos en zona frontal"""
        front_objects = []
        center_x = frame_width / 2

        for obj in objects:
            if obj.positions:
                last_pos = obj.positions[-1]
                cx = last_pos[0]
                # Zona central (40% del frame)
                if abs(cx - center_x) < frame_width * 0.2:
                    front_objects.append(obj)

        return front_objects

    def _get_objects_in_left(self, objects: List, frame_width: int, frame_height: int) -> List:
        """Obtener objetos en zona izquierda"""
        left_objects = []
        center_x = frame_width / 2

        for obj in objects:
            if obj.positions:
                last_pos = obj.positions[-1]
                cx = last_pos[0]
                # Zona izquierda
                if cx < center_x - frame_width * 0.1:
                    left_objects.append(obj)

        return left_objects

    def _get_objects_in_right(self, objects: List, frame_width: int, frame_height: int) -> List:
        """Obtener objetos en zona derecha"""
        right_objects = []
        center_x = frame_width / 2

        for obj in objects:
            if obj.positions:
                last_pos = obj.positions[-1]
                cx = last_pos[0]
                # Zona derecha
                if cx > center_x + frame_width * 0.1:
                    right_objects.append(obj)

        return right_objects

    def _get_average_risk(self, objects: List) -> float:
        """Obtener riesgo promedio de objetos"""
        if not objects:
            return 1.0  # LOW

        risk_values = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
        total_risk = 0
        count = 0

        for obj in objects:
            if obj.risk_levels:
                risk_val = risk_values.get(obj.risk_levels[-1], 1)
                total_risk += risk_val
                count += 1

        return total_risk / count if count > 0 else 1.0

    def _get_spanish_name(self, class_name: str) -> str:
        """Obtener nombre en español"""
        spanish_names = {
            'person': 'persona',
            'bicycle': 'bicicleta',
            'car': 'automóvil',
            'motorcycle': 'motocicleta',
            'bus': 'autobús',
            'truck': 'camión',
            'train': 'tren',
            'airplane': 'avión',
            'boat': 'bote',
            'dog': 'perro',
            'cat': 'gato',
            'horse': 'caballo',
            'cow': 'vaca',
            'elephant': 'elefante',
            'bear': 'oso',
        }
        return spanish_names.get(class_name, class_name)

    def _get_plural(self, singular_name: str) -> str:
        """Obtener forma plural"""
        plurals = {
            'persona': 'personas',
            'bicicleta': 'bicicletas',
            'automóvil': 'automóviles',
            'motocicleta': 'motocicletas',
            'autobús': 'autobuses',
            'camión': 'camiones',
            'tren': 'trenes',
            'avión': 'aviones',
            'bote': 'botes',
            'perro': 'perros',
            'gato': 'gatos',
            'caballo': 'caballos',
            'vaca': 'vacas',
            'elefante': 'elefantes',
            'oso': 'osos',
        }
        return plurals.get(singular_name, singular_name + 's')

    def _get_article(self, noun: str) -> str:
        """Obtener artículo correcto"""
        feminine_words = ['persona', 'bicicleta', 'motocicleta', 'vaca']
        if noun in feminine_words or noun.endswith(('a', 'ción', 'sión')):
            return 'una'
        else:
            return 'un'
