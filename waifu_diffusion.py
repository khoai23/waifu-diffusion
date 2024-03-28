"""Waifu-diffusion 1.5 beta; illusion variant; can be switched if necessary."""

from .base_model import BaseModelCkpt, BooruPromptGenerator 

ACCEPTABLE_MODE = {"illusion", "ink", "mofu", "radiance"}
MODEL_PATH = "https://huggingface.co/waifu-diffusion/wd-1-5-beta3/blob/main/wd-{:s}-fp16.safetensors"
VAE_PATH = "https://huggingface.co/hakurei/waifu-diffusion-v1-4/blob/main/vae/kl-f8-anime2.ckpt"

class WaifuDiffusionModel(BaseModelCkpt, BooruPromptGenerator):
    def __init__(self, mode: str="illusion", **kwargs):
        assert mode in ACCEPTABLE_MODE, "Cannot create model of mode \"{}\" for WaifuDiffusionModel".format(mode)
        path = MODEL_PATH.format(mode)
        super(WaifuDiffusionModel, self).__init__(path, vae_path=VAE_PATH, **kwargs)

        # in addition, save a bunch of tested stuff to allow querying
        self.template = {
            "attire": {
                "casual": (("({primary_color:s} coat, belt)", "({secondary_color:s} camisole, lace:2.0)", "(silver necklace)", "({secondary_color:s} {preferred_accessory:s})"), ("(exposed skin)", "(revealing clothes)")),
                "casual_skirt": (("({primary_color:s} sundress, long dress:1.2)", "({primary_color:s} hat, picture hat)", "({secondary_color:s} tanktop)", "({preferred_metal:s} necklace)"), ("schoolgirl", "student")),
                "dress_rich": (("({primary_color:s} dress, intricate dress, ornamented dress, {secondary_color:s} sleeves)", "({secondary_color:s} {preferred_accessory:s})"), ("(exposed skin)",)),
                "wedding_dress": (("(wedding dress, lace), bridal veil", "{primary_color:s} ribbon", "{secondary_color:s} bouquet"), ()),
                "armored": (("(battle dress: 1.2)", "(breastplate, helm, {preferred_metal:s} trim)", "(plate armor, {preferred_metal:s} ornament)", "({primary_color:s} skirt, long skirt)", ), ("bikini", "exposed")),
                "military": (("(military uniform: 2.0)", "({preferred_metal:s} trim)", "({primary_color:s} coat, {primary_color:s} shirt)", "({secondary_color:s} shirt, collared shirt)", "service cap"), ("exposed", "low-cut")),
                "cooking": (("({primary_color:s} sweater)", "({secondary_color:s} shirt, {secondary_color:s} pants, jeans)", "({secondary_color:s}-framed glasses, semi-rimless glasses)"), ("((shorts))", "((skirt))")),
            },
            "hair": {
                "long": (("(long hair, {hair_properties:s}, {hair_color:s} hair:2.0)",), ("short hair",)),
                "medium": (("(medium hair, {hair_properties:s}, {hair_color:s} hair:2.0)",), ("short hair", )),
                "short": (("(short hair, {hair_properties:s}, {hair_color:s} hair:2.0)",), ("long hair", )),
                # extra option to exclusively kill ornaments
                "no_ornament": ((), ("(hair ornament)",))
            },
            "eyes": {
                "default": (("({eye_color:s} eyes: 2.0, {eye_properties:s}, shaped pupils)",), ()),
                "closed": (("(closed eyes)", "blind", "(^ ^:2.0)"), ("(open eyes)",))
            },
            "expression": {
                "default": (("blush",), ()),
                "disdain": (("(grimace, disgust, scowl:2.5)", "(jitome:2.0, bored:1.2, averting eyes)"), ("smiling", "grinning", "((looking at viewer))")),
            },
            "context": {
                "default": (("{additional_properties:s}", "room:1.2"), ("looking at viewer",)),
                "outdoor": (("{additional_properties:s}", "park", "bench", "trees", "sunny"), ("cloudy", "night", "(plain background)")),
                "indoor": (("{additional_properties:s}", "room", "(wooden door)", "(leaning on wall)"), ("(plain background)",)),
                "cathedral": (("{additional_properties:s}", "stained glass", "cathedral", "marbled floor"), ("(plain background)",))
            },
            "pow": {
                "close-up": (("(close-up:0.5)", "(upper body:1.5)", "(solo focus)"), ())
            },
            "properties": {
                "type": (("1girl", "anime:2.0", "waifu:0.5"), ()),
                "quality": (("exceptional", "((best quality))",), ("lowres", "low quality", "worst quality")),
                "negative": ((), ("((bad anatomy))", "blurry", "(poorly drawn face)", "((mutation))", "((deformed face))", "(ugly)", "((bad proportions))", "monster", "logo", "cropped", "jpeg",
"(mosaic censoring, bar censor, blur censor)", "((extra limbs)), extra face, (double head), (extra head), ((extra feet)), (extra hand)")),
                "negative_hand": ((), ("((mutated hands and fingers))", "extra digits", "fewer digits"))
            }
        }
