# input/audio/audio_manager.py
import speech_recognition as sr
import audioop
from colorama import Fore

class AudioManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.energy_threshold = 300  # Nivel de energía mínima aceptable

    def calibrate_microphone(self, duration=2):
        """Calibra el micrófono al ruido ambiente antes de empezar."""
        with self.mic as source:
            print(Fore.YELLOW + f"[CONFIG] Calibrando micrófono ({duration}s, guarda silencio)...")
            self.recognizer.adjust_for_ambient_noise(source, duration=duration)
            self.energy_threshold = self.recognizer.energy_threshold
            print(Fore.GREEN + f"[SUCCESS] Micrófono calibrado. Umbral de energía: {self.energy_threshold:.2f}")

    def listen(self, timeout=6, phrase_time_limit=6):
        with self.mic as source:
            print(Fore.CYAN + "[INFO] Escuchando...")
            self.recognizer.energy_threshold = self.energy_threshold * 1.2
            self.recognizer.dynamic_energy_threshold = False
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                rms = audioop.rms(audio.get_raw_data(), 2)
                if rms < 100:
                    print(Fore.YELLOW + "[INFO] Ruido demasiado bajo, ignorando entrada.")
                    return None
                return audio
            except sr.WaitTimeoutError:
                print(Fore.RED + "[INFO] Tiempo de espera agotado.")
                return None
            except Exception as e:
                print(Fore.RED + f"[ERROR] Error al escuchar: {e}")
                return None

    def recognize(self, audio):
        if audio is None:
            return ""
        try:
            text = self.recognizer.recognize_google(audio, language="es-ES")  # type: ignore
            return text.lower().strip()
        except sr.UnknownValueError:
            print(Fore.YELLOW + "No se detectó voz clara.")
            return ""
        except sr.RequestError:
            print(Fore.RED + "Error de conexión con el servicio de reconocimiento.")
            return ""
        except Exception as e:
            print(Fore.RED + f"Error en reconocimiento: {e}")
            return ""
