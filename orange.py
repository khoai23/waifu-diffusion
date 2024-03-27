"""Orange-Mix model; requires further testing to create appropriate prompt for it."""

from .base_model import BaseModelCkpt, BooruPromptGenerator 

VAE_PATH = "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/VAEs/orangemix.vae.pt"
MODEL_PATH = "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/VividOrangeMix/VividOrangeMix.safetensors"

class GyozaModel(BaseModelCkpt, BooruPromptGenerator):
    def __init__(self, **kwargs):
        super(GyozaModel, self).__init__(MODEL_PATH, vae_path=VAE_PATH, **kwargs)
