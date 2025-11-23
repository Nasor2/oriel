#!/usr/bin/env python3
"""
Test script para verificar qué detecta el AlertSystem.
Muestra detecciones, análisis de riesgo y alertas en tiempo real.

Uso:
    python test_alert_system.py

El script:
1. Inicializa el AlertSystem en modo verbose
2. Captura 10 frames desde la cámara
3. Muestra detecciones y alertas para cada frame
4. Reporta estadísticas finales
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorama import Fore, init
init(autoreset=True)

from core.event_bus import EventBus
from input.video.video_manager import VideoManager
from visual.sai import AlertSystem


def test_alert_system():
    """Test del AlertSystem con visualización detallada."""

    print(Fore.CYAN + "=" * 80)
    print(Fore.CYAN + "TEST: AlertSystem - Detección y Alertas en Tiempo Real")
    print(Fore.CYAN + "=" * 80)

    # Inicializar
    event_bus = EventBus()

    print(Fore.YELLOW + "\n[SETUP] Inicializando VideoManager...")
    video_manager = VideoManager(source="http://192.168.1.96:8080/video")

    print(Fore.YELLOW + "[SETUP] Inicializando AlertSystem (modo verbose)...")
    alert_system = AlertSystem(
        event_bus=event_bus,
        video_manager=video_manager,
        mode="production",
        verbose=True  # ← Esto activa logging detallado
    )

    print(Fore.YELLOW + "[SETUP] Iniciando stream de alertas...")
    alert_system.start()

    print(Fore.GREEN + "\n[RUNNING] AlertSystem activo. Procesando frames...\n")

    # Mantener corriendo durante 30 segundos aprox
    # (El stream procesa en background, así que aquí solo esperamos)
    import time
    try:
        print(Fore.LIGHTBLACK_EX + "Ejecutando... (presiona Ctrl+C para detener)")
        time.sleep(30)
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n[INTERRUPTING] Usuario detuvo el test.")

    # Detener
    print(Fore.YELLOW + "\n[CLEANUP] Deteniendo AlertSystem...")
    alert_system.stop()

    # Estadísticas
    stats = alert_system.get_stats()
    print(Fore.CYAN + "\n" + "=" * 80)
    print(Fore.CYAN + "ESTADÍSTICAS FINALES")
    print(Fore.CYAN + "=" * 80)
    print(Fore.LIGHTGREEN_EX + f"Frames procesados: {stats['frames_processed']}")
    print(Fore.LIGHTGREEN_EX + f"Detecciones totales: {stats['detections_total']}")
    print(Fore.LIGHTGREEN_EX + f"Alertas emitidas: {stats['alerts_emitted']}")
    print(Fore.LIGHTGREEN_EX + f"Latencia promedio: {stats['inference_time_avg']:.1f} ms")

    if stats['frames_processed'] > 0:
        avg_det_per_frame = stats['detections_total'] / stats['frames_processed']
        print(Fore.LIGHTBLUE_EX + f"Promedio detecciones/frame: {avg_det_per_frame:.1f}")

    print(Fore.CYAN + "=" * 80 + "\n")


if __name__ == "__main__":
    try:
        test_alert_system()
    except KeyboardInterrupt:
        print(Fore.RED + "\n[INTERRUPTED] Test detenido por usuario.")
    except Exception as e:
        print(Fore.RED + f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

