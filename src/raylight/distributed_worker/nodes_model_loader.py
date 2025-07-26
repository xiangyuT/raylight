import folder_paths
from .flux.pipeline import (
    FluxMultiGPUPipeline,
    FluxSingleGPUPipeline,
    FluxT5EncoderFactory,
    FluxClipEncoderFactory,
    FluxTransfomerFactory,
    FluxVAEFactory
)

from .wan.pipeline import (
    WanMultiGPUPipeline,
    WanSingleGPUPipeline,
    WanT5EncoderFactory,
    WanClipEncoderFactory,
    WanTransfomerFactory,
    WanVAEFactory
)


class XFuserModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"),
                    {
                        "tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",
                    },
                ),
                "model_type": (
                    ["flux", "wan",],
                    {"default": "wan"},
                ),
                "base_precision": (
                    ["fp32", "bf16", "fp16", "fp16_fast"],
                    {"default": "bf16"},
                ),

            },
            "optional": {
                "ulysses_degree": ("INT", {"default": 1}),
                "ring_degree": ("INT", {"default": 1}),
                "use_fsdp": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "RayLight"

    def load_model(self, model, use_fsdp, t5_model_path, max_t5_token_length,
            use_xdit, ulysses_degree, ring_degree, cfg_parallel, model_type, base_precision):
        global num_gpus, pipeline, model_dir_path
        if pipeline is None:
            if model_type == "flux":
                FLUX_DIR = model
                print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
                klass = FluxSingleGPUPipeline if num_gpus == 1 else FluxMultiGPUPipeline
                kwargs = dict(
                    t5_text_encoder_factory=FluxT5EncoderFactory(
                        dtype=base_precision,
                    ),
                    clip_text_encoder_factory=FluxClipEncoderFactory(
                        model_path=f"{FLUX_DIR}/decoder.safetensors",
                        dtype=base_precision,
                    ),
                    dit_factory=FluxTransfomerFactory(
                        model_path=f"{FLUX_DIR}/dit.safetensors",
                        model_dtype="bf16",
                        dtype=base_precision,
                    ),
                    vae_factory=FluxVAEFactory(
                        model_path=f"{FLUX_DIR}/decoder.safetensors",
                        dtype=base_precision,
                    ),
                )

            if model_type == "wan":
                WAN_DIR = model
                print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
                klass = WanSingleGPUPipeline if num_gpus == 1 else WanMultiGPUPipeline
                kwargs = dict(
                    t5_text_encoder_factory=WanT5EncoderFactory(
                        dtype=base_precision,
                    ),
                    clip_text_encoder_factory=WanClipEncoderFactory(
                        model_path=f"{WAN_DIR}/decoder.safetensors",
                        dtype=base_precision,
                    ),
                    dit_factory=WanTransfomerFactory(
                        model_path=f"{WAN_DIR}/dit.safetensors",
                        model_dtype="bf16",
                        dtype=base_precision,
                    ),
                    vae_factory=WanVAEFactory(
                        model_path=f"{WAN_DIR}/decoder.safetensors",
                        dtype=base_precision,
                    ),
                )

        if num_gpus > 1:
            kwargs["world_size"] = num_gpus
            kwargs["use_xdit"] = use_xdit
            kwargs["ulysses_degree"] = ulysses_degree
            kwargs["ring_degree"] = ring_degree
            kwargs["cfg_parallel"] = cfg_parallel
        kwargs["use_fsdp"] = use_fsdp
        kwargs["t5_model_path"] = t5_model_path
        kwargs["max_t5_token_length"] = max_t5_token_length
        kwargs["decode_type"] = "tiled_full"
        pipeline = klass(**kwargs)


NODE_CLASS_MAPPINGS = {
    "XFuserModelLoader": XFuserModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XFuserModelLoader": "XFuser Model Loader",
}
