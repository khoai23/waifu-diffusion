"""RePaint logic; which claim to bring similar result to dedicated inpaint model except being slower & not too accurate around the masked region."""
from diffusers import StableDiffusionPipeline, RePaintScheduler
import os, io, PIL

class RepainterModelCkpt:
    def __init__(self, model_path: str, device: str="cuda"):
        # build & replace with appropriate scheduler as per tutorial
        model = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=torch.float16, custom_pipeline="stable_diffusion_repaint")
        model.scheduler = RePaintScheduler.from_config(model.scheduler.config) 
        # extra-length tokenizer to allow very large input
        self.tokenizer = Compel(tokenizer=model.tokenizer, text_encoder=model.text_encoder)

        self.model = model.to(device)
        self.device = device 


    def inpaint_image(self, prompt: str, negative_prompt: str, image: PIL.Image, mask_image: PIL.Image, guidance: float=7.5, num_inference_steps: int=200):
        conditioning = self.tokenizer.build_conditioning_tensor(prompt).to(self.device)
        negative_conditioning = self.tokenizer.build_conditioning_tensor(negative_prompt).to(self.device)
        new_image = self.model(prompt_embeds=conditioning, negative_prompt_embeds=negative_conditioning, image=image, mask_image=mask_image, num_inference_steps=num_inference_steps, guidance_scale=guidance)
        return new_image
