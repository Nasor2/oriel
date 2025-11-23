# input/audio/command_processor.py
import difflib
import spacy
import time
from functools import lru_cache
from colorama import Fore
import unicodedata

class CommandProcessor:
    """
    Interpreta comandos de voz con spaCy + coincidencia difusa.
    Maneja tildes, sinónimos, negaciones y stopwords relevantes.
    """

    EXTERNAL_INTENTS = [
        "disable_alert_system",
        "enable_alert_system",
        "footpath_request",
        "scene_description",
    ]

    INTERNAL_INTENTS = ["system_status"]

    def __init__(self):
        start = time.time()
        self.nlp = spacy.load("es_core_news_sm", disable=["ner", "parser"])
        print(Fore.CYAN + f"[NLP] Modelo spaCy cargado en {(time.time() - start)*1000:.1f} ms")

        self.commands = {
            "enable_alert_system": {
                "keywords": [
                    "activar sistema alerta",
                    "encender alerta",
                    "iniciar monitoreo",
                    "activar alarma",
                    "prender sistema alerta",
                ],
                "response": "Sistema de alertas activado.",
            },
            "disable_alert_system": {
                "keywords": [
                    "desactivar sistema alerta",
                    "apagar sistema alerta",
                    "detener alerta",
                    "quitar alerta",
                    "apagar alarma",
                    "deshabilitar sistema alerta",
                ],
                "response": "Sistema de alertas desactivado.",
            },
            "footpath_request": {
                "keywords": [
                    "caminar",
                    "por donde caminar",
                    "sendero",
                    "camino",
                    "hacia donde ir",
                    "ruta segura",
                    "por dónde camino",
                    "¿por dónde caminar?",
                    "¿estoy centrado?",
                    "¿la acera sigue?",
                    "¿tengo que girar?",
                    "¿hay camino?",
                    "¿por donde sigo?",
                    "dirección a seguir",
                    "ruta a seguir"
                ],
                "response": "Analizando el camino frente a ti.",
            },
            "scene_description": {
                "keywords": [
                    "que tengo enfrente",
                    "que hay delante",
                    "que ves",
                    "descríbeme",
                    "dime que hay",
                    "observa",
                    "analiza la escena",
                ],
                "response": "Procesando la escena frente a ti.",
            },
            "system_status": {
                "keywords": [
                    "estado",
                    "como estas",
                    "estas activo",
                    "sigues funcionando",
                    "sistema activo",
                    "estatus del sistema",
                    "funcionas",
                    "todo bien",
                    "todo correcto",
                    "todo okay",
                ],
                "response": "Estoy activo y escuchando.",
            },
        }

        self.important_words = {
            "alerta", "alarma", "activar", "desactivar", "apagar",
            "encender", "estado", "camino", "sendero", "escena",
            "bien", "todo", "como", "funcionas", "activo", "delante"
        }

    @staticmethod
    def _remove_accents(text: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    @lru_cache(maxsize=256)
    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = self._remove_accents(text)
        doc = self.nlp(text)
        lemmas = [
            t.lemma_ for t in doc
            if t.is_alpha and (not t.is_stop or t.lemma_ in self.important_words)
        ]
        return " ".join(lemmas)

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _detect_negation(self, text: str) -> bool:
        negation_terms = ["no", "apagar", "desactivar", "detener", "quitar", "deshabilitar"]
        return any(term in text for term in negation_terms)

    def process(self, text: str):
        if not text:
            return None, None, 0.0

        raw_text = text.lower().strip()
        raw_text_clean = self._remove_accents(raw_text)

        # FAST MATCH
        for intent, data in self.commands.items():
            for kw in data["keywords"]:
                if self._remove_accents(kw) in raw_text_clean:
                    response = data["response"]
                    print(Fore.MAGENTA + f"[FAST MATCH] '{text}' → '{kw}' | intención: {intent} (1.00)")
                    return intent, response, 1.0

        # Normalización + similitud difusa
        start = time.time()
        clean_text = self._normalize(text)
        best_intent, best_response, best_score = "unknown", "", 0.0

        for intent, data in self.commands.items():
            for kw in data["keywords"]:
                kw_clean = self._remove_accents(kw)
                score = self._similarity(clean_text, kw_clean)
                if kw_clean in clean_text:
                    score += 0.25
                if score > best_score:
                    best_score = score
                    best_intent = intent
                    best_response = data["response"]

        elapsed = (time.time() - start) * 1000
        print(Fore.MAGENTA + f"[NLP] '{text}' → '{clean_text}' | intención: {best_intent} ({best_score:.2f}) | tiempo: {elapsed:.2f} ms")

        # ajuste por negación
        if ("activar" in clean_text and self._detect_negation(clean_text)) or "desactivar" in clean_text or "apagar" in clean_text:
            best_intent = "disable_alert_system"
            best_response = self.commands["disable_alert_system"]["response"]

        if best_score < 0.55:
            return "unknown", "No entendí el comando. ¿Podrías repetirlo?", best_score

        return best_intent, best_response, min(best_score, 1.0)
