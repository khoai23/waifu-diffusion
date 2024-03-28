"""Base model for waifu generation mechanism on Google Colab. Should contain the appropriate model creation & prompt generation basing on a shared format."""
import requests
import torch

import os, io, PIL
from IPython.display import display, Image
import diffusers
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, StableDiffusionPipeline, StableDiffusionInpaintPipeline, AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType
from google.colab import files
import base64

from typing import Any, Optional, Dict, List

DEFAULT_WORK_DIR = "/content/drive/MyDrive/Work/StabDiff_Resource"
DRIVE_MOUNTED = [False]

class BaseModelCkpt:
    """Model with checkpoint file in safetensors & optional vae in ckpt."""
    def __init__(self, model_path: str, vae_path: str=None, safety_checker: bool=None, device: str="cuda"):
        # model
        if vae_path:
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16)
            model = StableDiffusionPipeline.from_single_file(model_path, vae=vae, safety_checker=safety_checker, torch_dtype=torch.float16)
        else:
            vae = None 
            model = StableDiffusionPipeline.from_single_file(model_path, safety_checker=safety_checker, torch_dtype=torch.float16)
        # extra-length tokenizer to allow very large input
        self.tokenizer = Compel(tokenizer=model.tokenizer, text_encoder=model.text_encoder)
        # move the model to associating devices
        self.model = model.to(device)
        self.device = device

    def __del__(self):
        """Enforce propagation of deletion to all related pytorch object"""
        del self.model 
        del self.tokenizer
        return super(BaseModelCkpt, self).__del__(self)


    def generate_images(self, prompt: str, negative_prompt: str, guidance: float=7.5, img_count: int=2, num_inference_steps: int=200):
        conditioning = self.tokenizer.build_conditioning_tensor(prompt).to(self.device)
        negative_conditioning = self.tokenizer.build_conditioning_tensor(negative_prompt).to(self.device)
        images = self.model(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, num_inference_steps=num_inference_steps, guidance_scale=guidance, num_images_per_prompt=img_count).images
        return images 

    def ipynb_display(self, image):
        """Use colab backend to display an image. Do not work outside."""
        bio = io.BytesIO()
        image.save(bio, format="png")
        display(Image(bio.getvalue(), format="png"))

    def save_to_drive(self, image, path: str, base: str=DEFAULT_WORK_DIR):
        if not DRIVE_MOUNTED[0]:
            print("@save_to_drive: Drive not mounted; mounting (will request user confirmation).")
            from google.colab import drive 
            drive.mount("/content/drive")
            DRIVE_MOUNTED[0] = True
        if not path.endswith(".png"):
            path = path + ".png"
        with io.open(os.path.join(base, path), "wb") as imgf:
            image.save(imgf, format="png")
        print("Saved image to \"{:s}\"".format(path))

    def test_default_generate(self):
        raise NotImplementedError

class BasePromptGenerator:
    def generate_prompts(self, composition: List[Dict], traits: List[List[str]], output_str: bool=True):
        """Must & should be implemented in a per-model basis"""
        raise NotImplementedError

class BooruPromptGenerator:
    """Generic prompt builder, conforming to generic booru format [ desc 1, (desc 2), (desc 3: 1.2) ].
    Should have variants to support non-weighting & non-bracketing options."""
    def generate_prompts(self, composition, traits, output_str=True):
        all_composition = dict()
        for c in composition:
            all_composition.update(c)
        positive_tags, negative_tags = [], []
        for piter, niter in traits:
            if isinstance(piter, str):
                piter = [piter]
            piter = [pt if "{" not in pt else pt.format(**all_composition) for pt in piter]
            positive_tags.extend(piter)
            if isinstance(niter, str):
                niter = [niter]
            niter = [nt if "{" not in nt else nt.format(**all_composition) for nt in niter]
            negative_tags.extend(niter)
        if output_str:
            return ", ".join(positive_tags), ", ".join(negative_tags)
        return positive_tags, negative_tags
