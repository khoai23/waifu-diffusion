"""Gyoza model; requires further testing to create appropriate prompt for it."""

from .base_model import BaseModelCkpt, BooruPromptGenerator 

MODEL_PATH = "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/LingesGyoza.safetensors"

class GyozaModel(BaseModelCkpt, BooruPromptGenerator):
    def __init__(self, **kwargs):
        super(GyozaModel, self).__init__(MODEL_PATH, vae_path=None, **kwargs)
