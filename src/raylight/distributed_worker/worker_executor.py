# This is where the worker is being run, we must use context (ctx)
# Read ray docs for further documentation.
# Manually control torch.distribute using torch.mp.spawn is pain, so ray is the solution
# Based on vLLM and XDit (XFuser) implementation
import os
import torch
import torch.distributed as dist


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
        use_xdit,
        ulysses_degree,
        ring_degree,
        cfg_parallel,
    ):
        # THIS IS IMPORTANT I KNOW GLOBAL IS A NO NO, BUT BOY
        # TORCH DIST WILL COMPLAIN
        # Internally python pickle an object to be distributed,
        # and it must see certain conf or var to work properly hence global it.
        self.model = None
        self.model_name = None
        self.run_fn = None
        set_global_config()
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
                device_id=self.device,
            )
        pg = dist.group.WORLD
        cp = cp_ctx[cfg["model_name"]]
        cp.set_cp_group(pg, list(range(world_size)), local_rank)
        distributed_kwargs = dict(local_rank=local_rank, device_id=device_id, world_size=world_size)
        self.world_size = world_size
        self.local_rank = local_rank
        self.decode_type = decode_type
        self.decode_args = decode_args or {}

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

        def load_model(self, load_fn, *args, **kwargs):
            model_name = kwargs.get("model_name")

            if self.model is not None and self.model_name == model_name:
                return f"Model already loaded: {type(self.model)}"

            with t(f"loading model: {model_name}"):
                self.model = load_fn(*args, **kwargs)
                self.model_name = model_name
                return f"Model loaded: {type(self.model)}"

        #self.tokenizer = model_tokenizer["cfg"]
        #with t("load_text_encoder"):
        #    self.text_encoder = text_encoder_factory.get_model(**distributed_kwargs)
        #with t("load_dit"):
        #    self.dit = dit_factory.get_model(**distributed_kwargs)
        #with t("load_vae"):
        #    self.decoder = decoder_factory.get_model(**distributed_kwargs)

        t.print_stats()

    def run(self, *, fn, **kwargs):
        return fn(self, **kwargs)
