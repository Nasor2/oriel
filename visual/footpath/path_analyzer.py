# visual/footpath/path_analyzer.py
# visual/footpath/path_analyzer.py
"""
Analizador de caminos basado en SODD-CA v3 Secure.
Procesa máscaras de caminería y extrae métricas estructuradas.
"""

import numpy as np
import cv2
import math
from sklearn.linear_model import LinearRegression
from colorama import Fore
from .config import SODD_CA_CONFIG, CONVERSION_FACTORS, SAFETY_THRESHOLDS

class PathAnalyzer:
    """
    Analizador de caminos con enfoque SODD-CA v3 Secure.
    """

    def __init__(self):
        self.alpha = CONVERSION_FACTORS["alpha"]
        self.beta = CONVERSION_FACTORS["beta"]
        print(Fore.CYAN + "[PathAnalyzer] Analizador de caminos inicializado.")

    def analyze(self, pred_mask, visualize=False):
        """
        Analiza una máscara de predicción para extraer métricas de caminería.

        Args:
            pred_mask: numpy array (256, 256) con probabilidades
            visualize: bool, si se debe generar visualización (no implementado aquí)

        Returns:
            dict con métricas estructuradas
        """
        # --- Normalización ---
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 1:
            pred_mask = pred_mask[..., 0]

        mask = (pred_mask > 0.5).astype(np.uint8)
        H, W = mask.shape

        total_pixels = H * W
        sidewalk_pixels = np.count_nonzero(mask)
        coverage_ratio = sidewalk_pixels / total_pixels

        # Si menos del 5% de la imagen es acera, asumimos que NO estamos en un camino.
        if coverage_ratio < 0.05:
            return {
                'direction_label': 'no_sidewalk',  # Etiqueta específica
                'confidence': 0.0,
                'path_continuity_ratio': 0.0,
                'distance_front_ft': 0.0,
                'distance_left_ft': 0.0,
                'distance_right_ft': 0.0,
                'angle_deg': 0.0,
                'curvature_deg': 0.0,
                'n_components': 0,
                'has_large_jumps': False,
                'mask_clipped': False,
                'valid_ratio': 0.0
            }

        # --- 1. Limpieza morfológica adaptativa ---
        ksize = max(3, int(min(H, W) * 0.015))
        kernel = np.ones((ksize, ksize), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

        # --- 1.1 Extra: cierre vertical pequeño para unir huecos pequeños ---
        kernel_v = np.ones((max(3, H//60), 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_v, iterations=1)

        # --- 1.2 Conectividad: filtrar componentes pequeñas y quedarse con la más grande ---
        num_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        if num_labels > 1:
            # stats[0] es fondo
            areas = stats[1:, cv2.CC_STAT_AREA]
            valid_idxs = [i+1 for i,a in enumerate(areas) if a > 0.015 * H * W]  # ignora islas pequeñas
            if len(valid_idxs) == 0:
                mask_main = np.zeros_like(mask_clean)
            else:
                largest_idx = valid_idxs[np.argmax([areas[i-1] for i in valid_idxs])]
                mask_main = (labels == largest_idx).astype(np.uint8)
        else:
            mask_main = mask_clean.copy()

        # --- 2. División por bandas horizontales ---
        bands = SODD_CA_CONFIG["bands"]
        step = H // bands
        centroids = []
        band_areas = []

        min_area_px = SODD_CA_CONFIG["min_area_px"]

        for i in range(bands):
            y0 = H - (i + 1) * step
            y1 = H - i * step if i < bands - 1 else H
            y0 = max(0, y0)
            y1 = min(H, y1)
            band = mask_main[y0:y1, :]

            area = int(band.sum())
            band_areas.append(area)
            if area > min_area_px:
                M = cv2.moments(band.astype(np.uint8))
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00']) + y0
                    centroids.append((cx, cy))
                else:
                    centroids.append(None)
            else:
                centroids.append(None)

        # --- 3. Interpolación de centroides faltantes ---
        for i in range(bands):
            if centroids[i] is None:
                lower = next((j for j in range(i - 1, -1, -1) if centroids[j] is not None), None)
                upper = next((j for j in range(i + 1, bands) if centroids[j] is not None), None)
                if lower is not None and upper is not None:
                    x_interp = int((centroids[lower][0] + centroids[upper][0]) / 2)
                    y_interp = int(H - (i + 0.5) * step)
                    centroids[i] = (x_interp, y_interp)

        pts = np.array([c for c in centroids if c is not None])
        valid_ratio = sum(c is not None for c in centroids) / bands

        if len(pts) < 2:
            # No hay suficiente información útil
            return {
                'direction_label': 'no_path',
                'confidence': float(valid_ratio),
                'path_continuity_ratio': 0.0,
                'distance_front_ft': 0.0,
                'distance_left_ft': 0.0,
                'distance_right_ft': 0.0,
                'angle_deg': 0.0,
                'curvature_deg': 0.0,
                'n_components': 0,
                'has_large_jumps': False,
                'mask_clipped': False
            }

        # --- 4. Regresión ponderada (dirección global) ---
        X = pts[:, 1].reshape(-1, 1)  # y (row)
        y = pts[:, 0]                 # x (col)
        weights = np.array([band_areas[i] for i in range(bands) if centroids[i] is not None], dtype=float)
        weights = weights / (weights.sum() + 1e-9)

        model = LinearRegression().fit(X, y, sample_weight=weights)
        slope = model.coef_[0]
        angle_deg = math.degrees(math.atan(slope))

        # --- 5. Continuidad vertical real ---
        gap_tolerance_rows = SODD_CA_CONFIG["gap_tolerance_rows"]
        walkable_rows = np.any(mask_main > 0, axis=1)
        continuity_rows = 0
        gaps = 0
        for i in range(H - 1, -1, -1):
            if walkable_rows[i]:
                continuity_rows += 1
                gaps = 0
            else:
                gaps += 1
                if gaps > gap_tolerance_rows:
                    break
        path_continuity_ratio = continuity_rows / max(H, 1)
        distance_front_ft = continuity_rows * self.beta

        # --- 5.1 Número de componentes ---
        n_labels_main, labels_main = cv2.connectedComponents(mask_main, connectivity=8)
        ncomponents_main = n_labels_main - 1  # sin fondo

        # --- 6. Análisis lateral (bottom section) ---
        bottom_section = mask_main[int(0.7 * H):, :]
        if np.any(bottom_section):
            xs = np.where(np.any(bottom_section > 0, axis=0))[0]
            left_clear = xs.min() * self.alpha if len(xs) > 0 else 0.0
            right_clear = (W - 1 - xs.max()) * self.alpha if len(xs) > 0 else 0.0
        else:
            left_clear = right_clear = 0.0

        # --- 7. Curvatura ---
        curvature = 0.0
        if len(pts) >= 4:
            mid = len(pts) // 2
            slope_top = (pts[-1][0] - pts[mid][0]) / (pts[-1][1] - pts[mid][1] + 1e-9)
            slope_bot = (pts[mid][0] - pts[0][0]) / (pts[mid][1] - pts[0][1] + 1e-9)
            curvature = math.degrees(math.atan(slope_top - slope_bot))

        # --- 7.1 Detección de saltos grandes ---
        large_jumps = 0
        centroid_vertical_gap_factor = SODD_CA_CONFIG["centroid_vertical_gap_factor"]
        if len(pts) > 1:
            y_diffs = np.abs(np.diff(pts[:, 1]))
            jump_thresh = step * (centroid_vertical_gap_factor + 0.5)
            large_jumps = int(np.sum(y_diffs > jump_thresh))

        # --- 8. Fusionar confianza y continuidad ---
        final_conf = 0.6 * valid_ratio + 0.4 * path_continuity_ratio

        # --- 9. Determinar etiqueta de dirección ---
        final_conf_threshold_uncertain = SODD_CA_CONFIG["final_conf_threshold_uncertain"]
        final_conf_threshold_no_path = SODD_CA_CONFIG["final_conf_threshold_no_path"]

        # Verificar condiciones de seguridad
        has_large_jumps = large_jumps > SAFETY_THRESHOLDS["max_jumps"]
        mask_clipped = self._is_mask_clipped(mask_main, H, W)

        # Si hay más de una componente conectada importante o saltos grandes -> no_path
        if ncomponents_main > SAFETY_THRESHOLDS["max_components"] or has_large_jumps:
            # penalización fuerte
            if (final_conf < final_conf_threshold_no_path or
                ncomponents_main > SAFETY_THRESHOLDS["max_components"]):
                dir_label = 'no_path'
            else:
                # incrementa incertidumbre si aún hay algo de confianza
                dir_label = 'uncertain'
        else:
            # Determinar label basado en angulo y curvatura
            if final_conf < final_conf_threshold_uncertain:
                dir_label = 'uncertain'
            else:
                if abs(angle_deg) < 6 and abs(curvature) < 8:
                    dir_label = "straight"
                elif angle_deg > 0 and curvature > 5:
                    dir_label = "curve_left"
                elif angle_deg < 0 and curvature < -5:
                    dir_label = "curve_right"
                elif angle_deg > 0:
                    dir_label = "veer_left"
                elif angle_deg < 0:
                    dir_label = "veer_right"
                else:
                    dir_label = "uncertain"

        # --- 10. Resultado estructurado ---
        return {
            'distance_front_ft': float(distance_front_ft),
            'distance_left_ft': float(left_clear),
            'distance_right_ft': float(right_clear),
            'angle_deg': float(angle_deg),
            'curvature_deg': float(curvature),
            'path_continuity_ratio': float(path_continuity_ratio),
            'confidence': float(final_conf),
            'direction_label': dir_label,
            'n_components': int(ncomponents_main),
            'has_large_jumps': bool(has_large_jumps),
            'mask_clipped': bool(mask_clipped),
            'valid_ratio': float(valid_ratio)
        }

    def _is_mask_clipped(self, mask, H, W):
        """
        Verifica si la máscara está cortada en los bordes.
        Más permisivo para evitar falsos positivos.
        """
        # Verificar bordes superior e inferior
        top_row = mask[0, :]
        bottom_row = mask[H-1, :]

        # Verificar bordes izquierdo y derecho
        left_col = mask[:, 0]
        right_col = mask[:, W-1]

        # Solo considerar como "cortada" si hay una cantidad significativa de píxeles activos
        # Requerir más del 25% del borde para considerarlo como "cortado"
        top_active = np.sum(top_row) > W * 0.25
        bottom_active = np.sum(bottom_row) > W * 0.25
        left_active = np.sum(left_col) > H * 0.25
        right_active = np.sum(right_col) > H * 0.25

        return top_active or bottom_active or left_active or right_active

    def visualize_analysis(self, pred_mask, analysis_result):
        """
        Visualize the mask analysis results.
        """
        import matplotlib.pyplot as plt

        # Process mask as in analyze method
        if pred_mask.ndim == 3 and pred_mask.shape[-1] == 1:
            pred_mask = pred_mask[..., 0]

        mask = (pred_mask > 0.5).astype(np.uint8)

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Show original mask
        axes[0].imshow(pred_mask, cmap='gray')
        axes[0].set_title('Predicted Mask')
        axes[0].axis('off')

        # Show binary mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Binary Mask (thresholded)')
        axes[1].axis('off')

        plt.suptitle(f"Direction: {analysis_result.get('direction_label', 'N/A')} | "
                     f"Confidence: {analysis_result.get('confidence', 0):.2f}")
        plt.show()
