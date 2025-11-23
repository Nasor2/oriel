# main.py
import sys
import os
import datetime
from colorama import Fore, init

init(autoreset=True)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.event_bus import EventBus
from core.orchestador import Orchestrator
from output.tts_manager import TTSManager
from input.audio.audio_pipeline import AudioPipeline
from input.audio.audio_manager import AudioManager
from input.video.video_manager import VideoManager
from visual.sai import AlertSystem
from visual.footpath.footpath_system import FootpathSystem

def calibrate_system():
    print(Fore.LIGHTGREEN_EX + "\n═══════════════════════════════════════════════")
    print(Fore.LIGHTGREEN_EX + "      [INICIO DEL SISTEMA DE ASISTENCIA ORIEL]  ")
    print(Fore.LIGHTGREEN_EX + "═══════════════════════════════════════════════")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(Fore.LIGHTBLUE_EX + f"[{now}] Inicializando módulos...")
    print(Fore.YELLOW + "[CALIBRACIÓN] Comprobando dispositivos de audio y cámara...")
    print(Fore.GREEN + "[OK] Micrófono disponible.")
    print(Fore.GREEN + "[OK] Cámara detectada.")
    print(Fore.LIGHTYELLOW_EX + "\n[CALIBRACIÓN] Iniciando calibración de micrófono...")
    audio_test = AudioManager()
    audio_test.calibrate_microphone(duration=2)
    print(Fore.GREEN + "[OK] Calibración de micrófono completada.\n")
    print(Fore.CYAN + "[INFO] Todo listo para iniciar el pipeline.\n")

def main():
    calibrate_system()
    event_bus = EventBus()
    orchestrator = Orchestrator(event_bus)
    tts = TTSManager(event_bus)

    # Crear VideoManager
    video_manager = VideoManager(source="http://192.168.1.96:8080/video")

    # Crear AlertSystem (aún no iniciado hasta recibir intent)
    alert_system = AlertSystem(
        event_bus=event_bus,
        video_manager=video_manager,
        mode="production",
        verbose=False
    )

    # Crear FootpathSystem
    footpath_system = FootpathSystem(event_bus)

    # Inyectar AlertSystem en Orchestrator
    orchestrator.set_video_manager(video_manager)
    orchestrator.set_alert_system(alert_system)
    orchestrator.set_footpath_system(footpath_system)

    # Crear AudioPipeline
    audio_pipeline = AudioPipeline(event_bus, wakeword="lucy", listen_timeout=4.0)

    print(Fore.MAGENTA + "\n[BOOT] Núcleo iniciado. Esperando eventos...\n")
    print(Fore.LIGHTBLACK_EX + "──────────────────────────────────────────────\n")

    # loop principal: audio_pipeline ya tiene run() bloqueante
    audio_pipeline.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(Fore.RED + "\n[INTERRUPCIÓN] Sistema detenido por el usuario.")
    except Exception as e:
        print(Fore.RED + f"\n[ERROR FATAL] {e}")
