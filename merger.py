"""Found from the internet that the checkpoints can be merged by AUTOMATIC1111; and there is an equal functionality with diffusers.examples.community.checkpoint_merger"""
import diffusers 
from diffusers import DiffusionPipeline 

from .base_model import BaseModelCkpt

from typing import Any, Optional, Dict, List

class MergeModel(BaseModelCkpt):
    def __init__(self, paths: List[str], device: str="cuda", **kwargs):
        base = paths[0]
        base_model = DiffusionPipeline.from_pretrained(base, custom_pipeline="checkpoint_merger")
        model = base_model.merge(paths, **kwargs)

        self.model = model.to(device)
        self.device = device
        
