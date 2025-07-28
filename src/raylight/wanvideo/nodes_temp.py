from .pipeline import WanClipEncoderFactory
import folder_paths
import torch
import comfy.model_management as mm


class LoadWanVideoClipTextEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("clip_vision") + folder_paths.get_filename_list("text_encoders"), {"tooltip": "These models are loaded from 'ComfyUI/models/clip_vision'"}),
                 "precision": (["fp16", "fp32", "bf16"],
                    {"default": "fp16"}
                ),
            },
            "optional": {
                "load_device": (["main_device", "offload_device"], {"default": "offload_device"}),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("wan_clip_vision", )
    FUNCTION = "load"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Loads Wan clip_vision model from 'ComfyUI/models/clip_vision'"

    def load(self, model_name, precision=torch.float16, load_device=torch.device("cpu")):
        precision = torch.float16
        model_path = folder_paths.get_full_path("clip_vision", model_name)
        clip_class = WanClipEncoderFactory(dtype=precision, model_path=model_path, model_dtype=torch.float16)
        model = clip_class.get_model(local_rank=0, device_id=load_device, world_size=2)

        return (model.model,)


NODE_CLASS_MAPPINGS = {
    "LoadWanVideoClipTextEncoder": LoadWanVideoClipTextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadWanVideoClipTextEncoder": "XFuser Wan Clip Loader",
}
