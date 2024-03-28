"""Gyoza model; requires further testing to create appropriate prompt for it."""

from .base_model import BaseModelCkpt, BooruPromptGenerator 

ALL_MODEL_OPTIONS = {
    "linges": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/LingesGyoza.safetensors",
    "store": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/storeBoughtGyozaMix_storeboughtgyozav10.safetensors",
    "flat": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/storeBoughtGyozaMix_flatgyozaafterdark.safetensors",
    "evening": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/storeBoughtGyozaMix_eveninggyozasnackv10.safetensors"
}

class GyozaModel(BaseModelCkpt, BooruPromptGenerator):
    def __init__(self, mode="store", **kwargs):
        super(GyozaModel, self).__init__(ALL_MODEL_OPTIONS[mode], vae_path=None, **kwargs)
