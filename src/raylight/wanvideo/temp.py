import torch
from comfy.utils import load_torch_file


class WanVideoModelLoader:
    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None, vace_model=None, fantasytalking_model=None, multitalk_model=None, usp=None):
        assert not (vram_management_args is not None and block_swap_args is not None), "Can't use both block_swap_args and vram_management_args at the same time"
        lora_low_mem_load = merge_loras = False
        if lora is not None:
            for l in lora:
                lora_low_mem_load = l.get("low_mem_load", False)
                merge_loras = l.get("merge_loras", True)

        transformer = None
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        gguf = False
        if model.endswith(".gguf"):
            if quantization != "disabled":
                raise ValueError("Quantization should be disabled when loading GGUF models.")
            quantization = "gguf"
            gguf = True
            merge_loras = False


        manual_offloading = True

        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)

        if not gguf:
            sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)
        else:
            from diffusers.models.model_loading_utils import load_gguf_checkpoint
            sd = load_gguf_checkpoint(model_path)

        if quantization == "disabled":
            if "scaled_fp8" in sd:
                quantization = "fp8_e4m3fn_scaled"
            else:
                for k, v in sd.items():
                    if isinstance(v, torch.Tensor):
                        if v.dtype == torch.float8_e4m3fn:
                            quantization = "fp8_e4m3fn"
                            break
                        elif v.dtype == torch.float8_e5m2:
                            quantization = "fp8_e5m2"
                            break

        if "scaled_fp8" in sd and quantization != "fp8_e4m3fn_scaled":
            raise ValueError("The model is a scaled fp8 model, please set quantization to 'fp8_e4m3fn_scaled'")

        first_key = next(iter(sd))
        if first_key.startswith("model.diffusion_model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.diffusion_model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        elif first_key.startswith("model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd
        if not "patch_embedding.weight" in sd:
            raise ValueError("Invalid WanVideo model selected")
        dim = sd["patch_embedding.weight"].shape[0]
        in_features = sd["blocks.0.self_attn.k.weight"].shape[1]
        out_features = sd["blocks.0.self_attn.k.weight"].shape[0]
        in_channels = sd["patch_embedding.weight"].shape[1]
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
        ffn2_dim = sd["blocks.0.ffn.2.weight"].shape[1]

        if in_channels in [36, 48]:
            model_type = "i2v"
        elif in_channels == 16:
            model_type = "t2v"
        elif "control_adapter.conv.weight" in sd:
            model_type = "t2v"

        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30

        model_variant = "14B"
        if model_type == "i2v" or model_type == "fl2v":
            if "480" in model or "fun" in model.lower() or "a2" in model.lower() or "540" in model:
                model_variant = "i2v_480"
            elif "720" in model:
                model_variant = "i2v_720"
        elif model_type == "t2v":
            model_variant = "14B"

        if dim == 1536:
            model_variant = "1_3B"

        TRANSFORMER_CONFIG= {
            "dim": dim,
            "in_features": in_features,
            "out_features": out_features,
            "ffn_dim": ffn_dim,
            "ffn2_dim": ffn2_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": 16,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "rope_func": "comfy",
            "inject_sample_info": True if "fps_embedding.weight" in sd else False,
            "add_ref_conv": True if "ref_conv.weight" in sd else False,
            "in_dim_ref_conv": sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            "add_control_adapter": True if "control_adapter.conv.weight" in sd else False,
        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG)
        transformer.eval()

        #USPs
        if usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in transformer.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            transformer.model.forward = types.MethodType(usp_dit_forward, self.model)
            sp_size = get_sequence_parallel_world_size()


        # Additional cond latents
        if "add_conv_in.weight" in sd:
            def zero_module(module):
                for p in module.parameters():
                    torch.nn.init.zeros_(p)
                return module
            inner_dim = sd["add_conv_in.weight"].shape[0]
            add_cond_in_dim = sd["add_conv_in.weight"].shape[1]
            attn_cond_in_dim = sd["attn_conv_in.weight"].shape[1]
            transformer.add_conv_in = torch.nn.Conv3d(add_cond_in_dim, inner_dim, kernel_size=transformer.patch_size, stride=transformer.patch_size)
            transformer.add_proj = zero_module(torch.nn.Linear(inner_dim, inner_dim))
            transformer.attn_conv_in = torch.nn.Conv3d(attn_cond_in_dim, inner_dim, kernel_size=transformer.patch_size, stride=transformer.patch_size)

        comfy_model = WanVideoModel(
            WanVideoModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )

        if not gguf:
            if "fp8_e4m3fn" in quantization:
                dtype = torch.float8_e4m3fn
            elif quantization == "fp8_e5m2":
                dtype = torch.float8_e5m2
            else:
                dtype = base_dtype
            params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation", "text_embedding", "adapter", "add"}
            if "scaled" in quantization:
                params_to_keep = {"patch_embedding", "modulation","norm", "bias"}
            #if lora is not None:
            #    transformer_load_device = device
            if not lora_low_mem_load:
                log.info("Using accelerate to load and assign model weights to device...")
                param_count = sum(1 for _ in transformer.named_parameters())
                pbar = ProgressBar(param_count)
                for name, param in tqdm(transformer.named_parameters(),
                        desc=f"Loading transformer parameters to {transformer_load_device}",
                        total=param_count,
                        leave=True):
                    dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                    if "patch_embedding" in name:
                        dtype_to_use = torch.float32
                    set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])
                    pbar.update(1)

        comfy_model.diffusion_model = transformer
        comfy_model.load_device = transformer_load_device

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
        patcher.model.is_patched = False

        if "scaled" in quantization:
            scale_weights = {}
            for k, v in sd.items():
                if k.endswith(".scale_weight"):
                    scale_weights[k] = v

        del sd

        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = quantization
        patcher.model["auto_cpu_offload"] = True if vram_management_args is not None else False

        if 'transformer_options' not in patcher.model_options:
            patcher.model_options['transformer_options'] = {}
        patcher.model_options["transformer_options"]["block_swap_args"] = block_swap_args

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)

        return (patcher,)
