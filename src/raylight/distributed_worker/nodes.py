import folder_paths


class XFuserModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list("unet_gguf")
                    + folder_paths.get_filename_list("diffusion_models"),
                    {
                        "tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",
                    },
                ),
                "base_precision": (
                    ["fp32", "bf16", "fp16", "fp16_fast"],
                    {"default": "bf16"},
                ),
                "quantization": (
                    [
                        "disabled",
                        "fp8_e4m3fn",
                        "fp8_e4m3fn_fast",
                        "fp8_e5m2",
                        "fp8_e4m3fn_fast_no_ffn",
                        "fp8_e4m3fn_scaled",
                    ],
                    {"default": "disabled", "tooltip": "optional quantization method"},
                ),
                "load_device": (
                    ["main_device", "offload_device"],
                    {
                        "default": "main_device",
                        "tooltip": "Initial device to load the model to.",
                    },
                ),
            },
            "optional": {
                "USP_Degree": ("INT", {"default": 1}),
                "ring_degree": ("INT", {"default": 1}),
                "FSDP": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "worker_wrapper"
    CATEGORY = "WanVideoWrapper"

    def load():
        return "XFUSER LOCS"


NODE_CLASS_MAPPINGS = {
    "XFuserModelLoader": XFuserModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserModelLoader": "XFuser Model Loader",
}
