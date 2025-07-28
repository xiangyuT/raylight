import os
import random
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Union, cast
import types

import numpy as np
import ray
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from safetensors.torch import load_file
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import T5Tokenizer
from .model.modules.t5 import T5EncoderModel, T5SelfAttention
from .model.modules.clip import CLIPModel
from .model.wan_video_vae import WanVideoVAE
from .model.globals import get_t5_model, get_max_t5_token_length, is_use_fsdp
from .model.globals import set_t5_model, set_max_t5_token_length, set_use_fsdp, set_use_xdit, set_usp_config
# from xfuser.core.distributed.parallel_state import (
#     get_classifier_free_guidance_world_size,
#     get_classifier_free_guidance_rank,
#     get_cfg_group
# )
import folder_paths


from comfy.utils import load_torch_file


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


def setup_fsdp_sync(model, device_id, *, param_dtype, auto_wrap_policy) -> FSDP:
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=device_id,
        sync_module_states=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


class ModelFactory(ABC):
    def __init__(self, dtype=torch.bfloat16, **kwargs):
        self.dtype = dtype
        self.kwargs = kwargs

    @abstractmethod
    def get_model(self, *, local_rank: int, device_id: Union[int, Literal["cpu"]], world_size: int) -> Any:
        if device_id == "cpu":
            assert world_size == 1, "CPU offload only supports single-GPU inference"


class WanT5EncoderFactory(ModelFactory):
    def __init__(
        self,
        model_path,
        model_dtype,
        dtype
    ):
        super().__init__(dtype=dtype, model_path=model_path, model_dtype=model_dtype)

    def get_model(self, *, local_rank, device_id, world_size):

        super().get_model(local_rank=local_rank, device_id=device_id, world_size=world_size)

        script_directory = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_directory, "configs", "T5_tokenizer")
        sd = load_torch_file(self.kwargs["model_path"], safe_load=True)
        if "token_embedding.weight" not in sd and "shared.weight" not in sd:
            raise ValueError("Invalid T5 text encoder model, this node expects the 'umt5-xxl' model")
        if "scaled_fp8" in sd:
            raise ValueError("Invalid T5 text encoder model, fp8 scaled is not supported by this node")

        if "shared.weight" in sd:
            converted_sd = {}

            for key, value in sd.items():
                # Handle encoder block patterns
                if key.startswith('encoder.block.'):
                    parts = key.split('.')
                    block_num = parts[2]

                    # Self-attention components
                    if 'layer.0.SelfAttention' in key:
                        if key.endswith('.k.weight'):
                            new_key = f"blocks.{block_num}.attn.k.weight"
                        elif key.endswith('.o.weight'):
                            new_key = f"blocks.{block_num}.attn.o.weight"
                        elif key.endswith('.q.weight'):
                            new_key = f"blocks.{block_num}.attn.q.weight"
                        elif key.endswith('.v.weight'):
                            new_key = f"blocks.{block_num}.attn.v.weight"
                        elif 'relative_attention_bias' in key:
                            new_key = f"blocks.{block_num}.pos_embedding.embedding.weight"
                        else:
                            new_key = key

                    # Layer norms
                    elif 'layer.0.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm1.weight"
                    elif 'layer.1.layer_norm' in key:
                        new_key = f"blocks.{block_num}.norm2.weight"

                    # Feed-forward components
                    elif 'layer.1.DenseReluDense' in key:
                        if 'wi_0' in key:
                            new_key = f"blocks.{block_num}.ffn.gate.0.weight"
                        elif 'wi_1' in key:
                            new_key = f"blocks.{block_num}.ffn.fc1.weight"
                        elif 'wo' in key:
                            new_key = f"blocks.{block_num}.ffn.fc2.weight"
                        else:
                            new_key = key
                    else:
                        new_key = key
                elif key == "shared.weight":
                    new_key = "token_embedding.weight"
                elif key == "encoder.final_layer_norm.weight":
                    new_key = "norm.weight"
                else:
                    new_key = key
                converted_sd[new_key] = value
            sd = converted_sd

        model = T5EncoderModel(
            text_len=512,
            dtype=self.kwargs["dtype"],
            device=None,
            state_dict=sd,
            tokenizer_path=tokenizer_path,
            quantization=self.kwargs["quantization"]
        )

        if world_size > 1:
            model = setup_fsdp_sync(
                model,
                device_id=device_id,
                param_dtype=self.dtype,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={
                        T5SelfAttention,
                    },
                ),
            )
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"), dtype=self.dtype)  # type: ignore
        return model.eval()


class WanClipEncoderFactory(ModelFactory):
    def __init__(self, *, dtype, model_path: str, model_dtype: str):
        super().__init__(dtype=dtype, model_path=model_path, model_dtype=model_dtype)

    def get_model(self, *, local_rank, device_id, world_size):
        super().get_model(local_rank=local_rank, device_id=device_id, world_size=world_size)

        state_dict = load_file(self.kwargs["model_path"])
        if "log_scale" not in state_dict:
            raise ValueError("Invalid CLIP model, this node expectes the 'open-clip-xlm-roberta-large-vit-huge-14' model")

        device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else "cpu"
        model = CLIPModel(dtype=self.dtype, device=device, state_dict=state_dict)

        return model


class WanDiTFactory(ModelFactory):
    def __init__(self, *, dtype, model_path: str, model_dtype: str, attention_mode: Optional[str] = None):
        attention_mode = self.kwargs.get("attention_mode", "sdpa")

        if "sage" == attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        super().__init__(
            dtype=dtype, model_path=model_path, model_dtype=model_dtype, attention_mode=attention_mode
        )
        self.model_path = model_path

    def get_model(self, *, local_rank, device_id, world_size):
        from .wanvideo.modules.model import WanModel

        gguf = False
        if self.model_path.endswith(".gguf"):
            if self.kwargs["quantization"] != "disabled":
                raise ValueError("Quantization should be disabled when loading GGUF models.")
            quantization = "gguf"
            gguf = True

        if not gguf:
            sd = load_torch_file(self.model_path, device=device_id, safe_load=True)
        else:
            from diffusers.models.model_loading_utils import load_gguf_checkpoint
            sd = load_gguf_checkpoint(self.model_path)

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

        if "patch_embedding.weight" not in sd:
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

        model: nn.Module = torch.nn.utils.skip_init(
            WanModel,
            dim=dim,
            in_features=in_features,
            out_features=out_features,
            ffn_dim=ffn_dim,
            ffn2_dim=ffn2_dim,
            eps=1e-06,
            freq_dim=256,
            in_dim=in_channels,
            model_type=model_type,
            out_dim=16,
            text_len=512,
            num_heads=num_heads,
            num_layers=num_layers,
            attention_mode=self.kwargs["attention_mode"],
            rope_func="comfy",
            inject_sample_info=True if "fps_embedding.weight" in sd else False,
            add_ref_conv=True if "ref_conv.weight" in sd else False,
            in_dim_ref_conv=sd["ref_conv.weight"].shape[1] if "ref_conv.weight" in sd else None,
            add_control_adapter=True if "control_adapter.conv.weight" in sd else False,
        )

        if self.kwargs["usp"]:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in model.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            model.model.forward = types.MethodType(usp_dit_forward, self.model)
            sp_size = get_sequence_parallel_world_size()


        if local_rank == 0 or not is_use_fsdp():
            # FSDP syncs weights from rank 0 to all other ranks
            model.load_state_dict(load_file(self.kwargs["model_path"]))

        if world_size > 1:
            assert self.kwargs["model_dtype"] == "bf16", "FP8 is not supported for multi-GPU inference"
            if is_use_fsdp():
                model = setup_fsdp_sync(
                    model,
                    device_id=device_id,
                    param_dtype=torch.bfloat16,
                    auto_wrap_policy=partial(
                        lambda_auto_wrap_policy,
                        lambda_fn=lambda m: m in model.blocks,
                    ),
                )
            else:
                model = model.to(torch.device(f"cuda:{device_id}"), dtype=self.dtype)
        elif isinstance(device_id, int):
            model = model.to(torch.device(f"cuda:{device_id}"), dtype=self.dtype)
        return model.eval()


class WanVAEFactory(ModelFactory):
    def __init__(self, *, dtype, model_path: str):
        super().__init__(dtype=dtype, model_path=model_path)

    def get_model(self, *, local_rank, device_id, world_size):
        state_dict = load_file(self.kwargs["model_path"])

        has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())
        if not has_model_prefix:
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        vae = WanVideoVAE(dtype=self.dtype)

        vae.load_state_dict(state_dict, strict=True)

        device = torch.device(f"cuda:{device_id}") if isinstance(device_id, int) else "cpu"
        vae.eval().to(device, dtype=self.dtype)
        return vae


def get_conditioning(tokenizer, encoder, device, batch_inputs, *, prompt: str, negative_prompt: str):
    if batch_inputs:
        return dict(batched=get_conditioning_for_prompts(tokenizer, encoder, device, [prompt, negative_prompt]))
    else:
        cond_input = get_conditioning_for_prompts(tokenizer, encoder, device, [prompt])
        null_input = get_conditioning_for_prompts(tokenizer, encoder, device, [negative_prompt])
        return dict(cond=cond_input, null=null_input)


def get_conditioning_for_prompts(tokenizer, encoder, device, prompts: List[str]):
    assert len(prompts) in [1, 2]  # [neg] or [pos] or [pos, neg]
    B = len(prompts)
    t5_toks = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=get_max_t5_token_length(),
        return_tensors="pt",
        return_attention_mask=True,
    )
    caption_input_ids_t5 = t5_toks["input_ids"]
    caption_attention_mask_t5 = t5_toks["attention_mask"].bool()
    del t5_toks

    assert caption_input_ids_t5.shape == (B, get_max_t5_token_length())
    assert caption_attention_mask_t5.shape == (B, get_max_t5_token_length())

    # Special-case empty negative prompt by zero-ing it
    if prompts[-1] == "":
        caption_input_ids_t5[-1] = 0
        caption_attention_mask_t5[-1] = False

    caption_input_ids_t5 = caption_input_ids_t5.to(device, non_blocking=True)
    caption_attention_mask_t5 = caption_attention_mask_t5.to(device, non_blocking=True)

    y_mask = [caption_attention_mask_t5]
    y_feat = [encoder(caption_input_ids_t5, caption_attention_mask_t5).last_hidden_state.detach()]
    # Sometimes returns a tensor, othertimes a tuple, not sure why
    # See: https://huggingface.co/genmo/mochi-1-preview/discussions/3
    assert tuple(y_feat[-1].shape) == (B, get_max_t5_token_length(), 4096)
    # assert y_feat[-1].dtype == torch.float32

    return dict(y_mask=y_mask, y_feat=y_feat)


def compute_packed_indices(
    device: torch.device, text_mask: torch.Tensor, num_latents: int
) -> Dict[str, Union[torch.Tensor, int]]:
    """
    Based on https://github.com/Dao-AILab/flash-attention/blob/765741c1eeb86c96ee71a3291ad6968cfbf4e4a1/flash_attn/bert_padding.py#L60-L80

    Args:
        num_latents: Number of latent tokens
        text_mask: (B, L) List of boolean tensor indicating which text tokens are not padding.

    Returns:
        packed_indices: Dict with keys for Flash Attention:
            - valid_token_indices_kv: up to (B * (N + L),) tensor of valid token indices (non-padding)
                                   in the packed sequence.
            - cu_seqlens_kv: (B + 1,) tensor of cumulative sequence lengths in the packed sequence.
            - max_seqlen_in_batch_kv: int of the maximum sequence length in the batch.
    """
    # Create an expanded token mask saying which tokens are valid across both visual and text tokens.
    PATCH_SIZE = 2
    num_visual_tokens = num_latents // (PATCH_SIZE**2)
    assert num_visual_tokens > 0

    mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)  # (B, N + L)
    seqlens_in_batch = mask.sum(dim=-1, dtype=torch.int32)  # (B,)
    if not is_use_xdit():
        valid_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()  # up to (B * (N + L),)
        assert valid_token_indices.size(0) >= text_mask.size(0) * num_visual_tokens  # At least (B * N,)
    else:
        valid_token_indices = torch.nonzero(text_mask.flatten(), as_tuple=False).flatten()  # up to (B * L),)

    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen_in_batch = seqlens_in_batch.max().item()

    return {
        "cu_seqlens_kv": cu_seqlens.to(device, non_blocking=True),
        "max_seqlen_in_batch_kv": cast(int, max_seqlen_in_batch),
        "valid_token_indices_kv": valid_token_indices.to(device, non_blocking=True),
    }


def assert_eq(x, y, msg=None):
    assert x == y, f"{msg or 'Assertion failed'}: {x} != {y}"


def sample_model(device, dit, conditioning, **args):
    dtype = conditioning["cond"]["y_feat"][0].dtype
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert_eq(len(cfg_schedule), sample_steps, "cfg_schedule must have length sample_steps")
    assert_eq((t - 1) % 6, 0, "t - 1 must be divisible by 6")
    assert_eq(
        len(sigma_schedule),
        sample_steps + 1,
        "sigma_schedule must have length sample_steps + 1",
    )

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

    z = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=dtype,
    )

    num_latents = latent_t * latent_h * latent_w
    cond_batched = cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
        cond_text["packed_indices"] = compute_packed_indices(device, cond_text["y_mask"][0], num_latents)
        cond_null["packed_indices"] = compute_packed_indices(device, cond_null["y_mask"][0], num_latents)
    else:
        cond_batched = conditioning["batched"]
        cond_batched["packed_indices"] = compute_packed_indices(device, cond_batched["y_mask"][0], num_latents)
        z = repeat(z, "b ... -> (repeat b) ...", repeat=2)

    def model_fn(*, z, sigma, cfg_scale):
        if cond_batched:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = dit(z, sigma, **cond_batched)
            out_cond, out_uncond = torch.chunk(out, chunks=2, dim=0)
        else:
            nonlocal cond_text, cond_null
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if is_use_xdit() and get_classifier_free_guidance_world_size() == 2:
                    if get_classifier_free_guidance_rank() == 0:
                        out = dit(z, sigma, **cond_text)
                    else:
                        out = dit(z, sigma, **cond_null)
                    out_cond, out_uncond = get_cfg_group().all_gather(
                        out, separate_tensors=True
                    )
                else:
                    out_cond = dit(z, sigma, **cond_text)
                    out_uncond = dit(z, sigma, **cond_null)
        assert out_cond.shape == out_uncond.shape
        out_uncond = out_uncond.to(z)
        out_cond = out_cond.to(z)
        return out_uncond + cfg_scale * (out_cond - out_uncond)

    # Euler sampler w/ customizable sigma schedule & cfg scale
    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        # `pred` estimates `z_0 - eps`.
        pred = model_fn(
            z=z,
            sigma=torch.full([B] if cond_text else [B * 2], sigma, device=z.device, dtype=z.dtype),
            cfg_scale=cfg_schedule[i],
        )
        # assert pred.dtype == torch.float32
        z = z + dsigma * pred

    z = z[:B] if cond_batched else z
    return dit_latents_to_vae_latents(z)


@contextmanager
def move_to_device(model: nn.Module, target_device):
    og_device = next(model.parameters()).device
    if og_device == target_device:
        print(f"move_to_device is a no-op model is already on {target_device}")
    else:
        print(f"moving model from {og_device} -> {target_device}")

    model = model.to(target_device)
    yield
    if og_device != target_device:
        print(f"moving model from {target_device} -> {og_device}")
    model = model.to(og_device)

class WanSingleGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        cpu_offload: Optional[bool] = False,
        decode_type: str = "full",
        decode_args: Optional[Dict[str, Any]] = None,
        use_fsdp,
        t5_model_path,
        max_t5_token_length,
    ):
        set_use_fsdp(use_fsdp)
        set_t5_model(t5_model_path)
        set_max_t5_token_length(max_t5_token_length)

        self.device = torch.device("cuda:0")
        self.tokenizer = t5_tokenizer()
        t = Timer()
        self.cpu_offload = cpu_offload
        self.decode_args = decode_args or {}
        self.decode_type = decode_type
        init_id = "cpu" if cpu_offload else 0
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(
                local_rank=0,
                device_id=init_id,
                world_size=1,
            )
        with t("load_dit"):
            self.dit = dit_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(local_rank=0, device_id=init_id, world_size=1)
        t.print_stats()

    def __call__(self, batch_cfg, prompt, negative_prompt, **kwargs):
        with torch.inference_mode():
            print_max_memory = lambda: print(
                f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB"
            )
            print_max_memory()

            t = Timer()
            with t("conditioning"), move_to_device(self.text_encoder, self.device):
                conditioning = get_conditioning(
                    self.tokenizer,
                    self.text_encoder,
                    self.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            print_max_memory()

            with t("sampling"), move_to_device(self.dit, self.device):
                latents = sample_model(self.device, self.dit, conditioning, **kwargs)
            print_max_memory()

            with t("decoding"), move_to_device(self.decoder, self.device):
                frames = (
                    decode_latents_tiled_full(self.decoder, latents, **self.decode_args)
                    if self.decode_type == "tiled_full"
                    else decode_latents_tiled_spatial(self.decoder, latents, **self.decode_args)
                    if self.decode_type == "tiled_spatial"
                    else decode_latents(self.decoder, latents)
                )
            print_max_memory()

            t.print_stats()  # Print timing statistics at the end
            return frames.cpu().numpy()

# == ALL CODE BELOW HERE IS FOR MULTI-GPU MODE == #


# In multi-gpu mode, all models must belong to a device which has a predefined context parallel group
# So it doesn't make sense to work with models individually
class MultiGPUContext:
    def __init__(
        self,
        *,
        text_encoder_factory,
        dit_factory,
        decoder_factory,
        device_id,
        local_rank,
        world_size,
        decode_type,
        decode_args,
        use_fsdp,
        t5_model_path,
        max_t5_token_length,
        use_xdit,
        ulysses_degree,
        ring_degree,
        cfg_parallel,
    ):
        set_use_fsdp(use_fsdp)
        set_t5_model(t5_model_path)
        set_max_t5_token_length(max_t5_token_length)
        set_use_xdit(use_xdit)
        set_usp_config(ulysses_degree, ring_degree, cfg_parallel)

        t = Timer()
        self.device = torch.device(f"cuda:{device_id}")
        print(f"Initializing rank {local_rank+1}/{world_size}")
        assert world_size > 1, f"Multi-GPU mode requires world_size > 1, got {world_size}"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        with t("init_process_group"):
            dist.init_process_group(
                "nccl",
                rank=local_rank,
                world_size=world_size,
                device_id=self.device,  # force non-lazy init
            )
        pg = dist.group.WORLD
        cp.set_cp_group(pg, list(range(world_size)), local_rank)
        distributed_kwargs = dict(local_rank=local_rank, device_id=device_id, world_size=world_size)
        self.world_size = world_size
        self.local_rank = local_rank
        self.decode_type = decode_type
        self.decode_args = decode_args or {}

        # TODO(jiaruifang) confuse local_rank and rank, not applied to multi-node
        if is_use_xdit():
            cp_rank, cp_size = cp.get_cp_rank_size()
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )

            ulysses_degree, ring_degree, cfg_parallel = get_usp_config()
            init_distributed_environment(rank=cp_rank, world_size=cp_size)
            if not cfg_parallel:
                if ulysses_degree is None and ring_degree is None:
                    print(f"No usp config, use default config: ulysses_degree={cp_size}, ring_degree=1, CFG parallel false")
                    initialize_model_parallel(
                        sequence_parallel_degree=world_size,
                        ring_degree=1,
                        ulysses_degree=cp_size,
                    )
                else:
                    if ulysses_degree is None:
                        ulysses_degree = world_size // ring_degree
                    if ring_degree is None:
                        ring_degree = world_size // ulysses_degree
                    print(f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, CFG parallel false")
                    initialize_model_parallel(
                        sequence_parallel_degree=world_size,
                        ring_degree=ring_degree,
                        ulysses_degree=ulysses_degree,
                    )
            else:
                if ulysses_degree is None and ring_degree is None:
                    print(f"No usp config, use default config: ulysses_degree={cp_size // 2}, ring_degree=1, CFG parallel true")
                    initialize_model_parallel(
                        sequence_parallel_degree=world_size // 2,
                        ring_degree=1,
                        ulysses_degree=cp_size // 2,
                        classifier_free_guidance_degree=2,
                    )
                else:
                    if ulysses_degree is None:
                        ulysses_degree = world_size // 2 // ring_degree
                    if ring_degree is None:
                        ring_degree = world_size // 2 // ulysses_degree
                    print(f"Use usp config: ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, CFG parallel true")
                    initialize_model_parallel(
                        sequence_parallel_degree=world_size // 2,
                        ring_degree=ring_degree,
                        ulysses_degree=ulysses_degree,
                        classifier_free_guidance_degree=2,
                    )

        self.tokenizer = t5_tokenizer()
        with t("load_text_encoder"):
            self.text_encoder = text_encoder_factory.get_model(**distributed_kwargs)
        with t("load_dit"):
            self.dit = dit_factory.get_model(**distributed_kwargs)
        with t("load_vae"):
            self.decoder = decoder_factory.get_model(**distributed_kwargs)

        t.print_stats()

    def run(self, *, fn, **kwargs):
        return fn(self, **kwargs)


class WanMultiGPUPipeline:
    def __init__(
        self,
        *,
        text_encoder_factory: ModelFactory,
        dit_factory: ModelFactory,
        decoder_factory: ModelFactory,
        world_size: int,
        decode_type: str = "full",
        decode_args: Optional[Dict[str, Any]] = None,
        use_fsdp,
        t5_model_path,
        max_t5_token_length,
        use_xdit,
        ulysses_degree,
        ring_degree,
        cfg_parallel,
    ):
        ray.init()
        RemoteClass = ray.remote(MultiGPUContext)
        self.ctxs = [
            RemoteClass.options(num_gpus=1).remote(
                text_encoder_factory=text_encoder_factory,
                dit_factory=dit_factory,
                decoder_factory=decoder_factory,
                world_size=world_size,
                device_id=0,
                local_rank=i,
                decode_type=decode_type,
                decode_args=decode_args,
                use_fsdp=use_fsdp,
                t5_model_path=t5_model_path,
                max_t5_token_length=max_t5_token_length,
                use_xdit=use_xdit,
                ulysses_degree=ulysses_degree,
                ring_degree=ring_degree,
                cfg_parallel=cfg_parallel,
            )
            for i in range(world_size)
        ]
        for ctx in self.ctxs:
            ray.get(ctx.__ray_ready__.remote())

    def __call__(self, **kwargs):
        def sample(ctx, *, batch_cfg, prompt, negative_prompt, **kwargs):

            def print_max_memory():
                print(f"Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

            print_max_memory()

            t = Timer()
            with t("conditioning"), progress_bar(type="ray_tqdm", enabled=ctx.local_rank == 0), torch.inference_mode():
                conditioning = get_conditioning(
                    ctx.tokenizer,
                    ctx.text_encoder,
                    ctx.device,
                    batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
            print_max_memory()

            with t("sampling"), progress_bar(type="ray_tqdm", enabled=ctx.local_rank == 0), torch.inference_mode():
                latents = sample_model(ctx.device, ctx.dit, conditioning=conditioning, **kwargs)
            print_max_memory()

            if ctx.local_rank == 0:
                torch.save(latents, "latents.pt")

            with t("decoding"), progress_bar(type="ray_tqdm", enabled=ctx.local_rank == 0), torch.inference_mode():
                frames = (
                    decode_latents_tiled_full(ctx.decoder, latents, **ctx.decode_args)
                    if ctx.decode_type == "tiled_full"
                    else decode_latents_tiled_spatial(ctx.decoder, latents, **ctx.decode_args)
                    if ctx.decode_type == "tiled_spatial"
                    else decode_latents(ctx.decoder, latents)
                )
                print_max_memory()

            if ctx.local_rank == 0:
                t.print_stats()

            return frames.cpu().numpy()

        return ray.get([ctx.run.remote(fn=sample, **kwargs, show_progress=i == 0) for i, ctx in enumerate(self.ctxs)])[
            0
        ]
