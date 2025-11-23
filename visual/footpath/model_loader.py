# visual/footpath/model_loader.py
import os
import numpy as np
import tensorflow as tf
from colorama import Fore
from .config import MODEL_CONFIG

# Define the custom loss function with proper registration
def bce_dice_loss(y_true, y_pred):
    """Binary Cross-Entropy + Dice Loss"""
    # BCE component
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Dice component
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)
    dice_loss = 1 - dice

    # Combined loss
    return bce + dice_loss

class FootpathModelLoader:
    """
    Carga y gestiona el modelo de segmentación de caminerías.
    """

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carga el modelo .keras desde el path especificado."""
        try:
            model_path = MODEL_CONFIG["model_path"]
            if not os.path.isabs(model_path):
                model_path = os.path.join(os.getcwd(), model_path)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

            # Load model with custom objects using the correct registration
            custom_objects = {"bce_dice_loss": bce_dice_loss}
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False  # Load without compiling
            )

            # Compile model manually if needed
            self.model.compile(
                optimizer='adam',
                loss=bce_dice_loss
            )

            print(Fore.GREEN + f"[FootpathModel] Modelo cargado desde: {model_path}")
        except Exception as e:
            print(Fore.RED + f"[FootpathModel] Error al cargar modelo: {e}")
            raise

    def predict(self, image_tensor):
        """
        Realiza inferencia con el modelo.

        Args:
            image_tensor: numpy array de forma (1, 256, 256, 3)

        Returns:
            mask: numpy array de forma (256, 256) con probabilidades
        """
        if self.model is None:
            raise RuntimeError("Modelo no cargado")

        try:
            # Realizar predicción
            pred = self.model.predict(image_tensor, verbose=0)[0]

            # Asegurar forma correcta
            if pred.ndim == 3 and pred.shape[-1] == 1:
                pred = pred[..., 0]

            # Normalizar a rango [0, 1]
            pred = np.clip(pred, 0, 1)
            return pred
        except Exception as e:
            print(Fore.RED + f"[FootpathModel] Error en inferencia: {e}")
            raise
