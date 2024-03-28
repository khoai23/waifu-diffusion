"""Gyoza model; requires further testing to create appropriate prompt for it."""

from .base_model import BaseModelCkpt, BooruPromptGenerator  
from .waifu_diffusion import TEMPLATE as OLD_TEMPLATE

TEMPLATE = dict(OLD_TEMPLATE)
TEMPLATE["properties"] = {
    "quality": (("(masterpiece, best quality, extremely detailed:1.4)", ) ("(Worst Quality, Low Quality:1.4), negfeet",)),
    "break": (("BREAK", ), ()),
    "type": (("1girl", "solo"), ())
}

ALL_MODEL_OPTIONS = {
    "mdl_linges": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/LingesGyoza.safetensors",
    "mdl_store": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/storeBoughtGyozaMix_storeboughtgyozav10.safetensors",
    "mdl_flat": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/storeBoughtGyozaMix_flatgyozaafterdark.safetensors",
    "mdl_evening": "https://huggingface.co/mdl-mirror/Store-Bought-Gyoza/blob/main/storeBoughtGyozaMix_eveninggyozasnackv10.safetensors",
    "default": "https://huggingface.co/Jemnite/GyoZanMix/blob/main/factorygyozan/StoreGyoza.safetensors",
    "v2": "https://huggingface.co/Jemnite/GyoZanMix/blob/main/factorygyozan/GyozaMixV2.safetensors",
    "v3": "https://huggingface.co/Jemnite/GyoZanMix/blob/main/factorygyozan/GyozaMixV3fp16.safetensors",
    "v3.2": "https://huggingface.co/Jemnite/GyoZanMix/blob/main/factorygyozan/GyozaMixV3fp16.safetensors",
    "v3.5": "https://huggingface.co/Jemnite/GyoZanMix/blob/main/factorygyozan/GyozaMixV3.5.safetensors",
    "v5": "https://huggingface.co/Jemnite/GyoZanMix/blob/main/factorygyozan/GyozaMixV5.safetensors"
}

class GyozaModel(BaseModelCkpt, BooruPromptGenerator):
    def __init__(self, mode="v3.2", **kwargs):
        super(GyozaModel, self).__init__(ALL_MODEL_OPTIONS[mode], vae_path=None, **kwargs)

        self.template = TEMPLATE
