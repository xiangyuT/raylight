import folder_paths
from .flux.pipeline import FluxMultiGPUPipeline, FluxSingleGPUPipeline


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
            use_xdit, ulysses_degree, ring_degree, cfg_parallel, model_type):
        global num_gpus, pipeline, model_dir_path
        if pipeline is None:
            if model_type == "flux":
                FLUX_DIR = model
                print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
                klass = FluxSingleGPUPipeline if num_gpus == 1 else FluxMultiGPUPipeline
                kwargs = dict(
                    text_encoder_factory=T5ModelFactory(
                        dtype=dtype,
                    ),
                    dit_factory=DitModelFactory(
                        model_path=f"{MOCHI_DIR}/dit.safetensors",
                        model_dtype="bf16",
                        dtype=dtype,
                    ),
                    decoder_factory=DecoderModelFactory(
                        model_path=f"{MOCHI_DIR}/decoder.safetensors",
                        dtype=dtype,
                    ),
                )
            elif model_type == "wan":
                MOCHI_DIR = model_dir_path
                print(f"Launching with {num_gpus} GPUs. If you want to force single GPU mode use CUDA_VISIBLE_DEVICES=0.")
                klass = MochiSingleGPUPipeline if num_gpus == 1 else MochiMultiGPUPipeline
                kwargs = dict(
                    text_encoder_factory=T5ModelFactory(
                        dtype=dtype,
                    ),
                    dit_factory=DitModelFactory(
                        model_path=f"{MOCHI_DIR}/dit.safetensors",
                        model_dtype="bf16",
                        dtype=dtype,
                    ),
                    decoder_factory=DecoderModelFactory(
                        model_path=f"{MOCHI_DIR}/decoder.safetensors",
                        dtype=dtype,
                    ),
                )
        if num_gpus > 1:
            assert not cpu_offload, "CPU offload not supported in multi-GPU mode"
            kwargs["world_size"] = num_gpus
            kwargs["use_xdit"] = use_xdit
            kwargs["ulysses_degree"] = ulysses_degree
            kwargs["ring_degree"] = ring_degree
            kwargs["cfg_parallel"] = cfg_parallel
        else:
            kwargs["cpu_offload"] = cpu_offload
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
