# This is where the worker is being instantiated
# Read ray docs for further documentation.
# Manually control torch.distribute using torch.mp.spawn is pain, so ray is the solution
# Based on vLLM and XDit (XFuser) implementation
import os
import ray
import torch
from torch import distributed as dist
import psutil


class WorkerContext:
    def __init__(self, *, device_id, local_rank, world_size, ulysses_degree):
        self.device = torch.device(f"cpu:{device_id}")
        self.ulysses_degree = ulysses_degree
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group(
            "cuda",
            rank=local_rank,
            world_size=world_size,
            device_id=self.device
        )

        # This is for model persistency test, just delete this to load the actual model
        self.tensors = [torch.rand(256, 256) for _ in range(1000)]

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024**2
        return f"Resident memory: {mem_mb:.2f} MB, Tensor count: {len(self.tensors)}"

    def run(self, *, fn, **kwargs):
        return fn(self, **kwargs)


class SimpleWorkerPipeline:
    def __init__(self, *, world_size, ulysses_degree):
        ray.init(ignore_reinit_error=True)
        RemoteClass = ray.remote(WorkerContext)
        self.ctxs = [
            RemoteClass.options(num_cpus=1).remote(
                device_id=0,
                local_rank=i,
                world_size=world_size,
                ulysses_degree=ulysses_degree
            )
            for i in range(world_size)
        ]

        for ctx in self.ctxs:
            ray.get(ctx.__ray_ready__.remote())

    def __call__(self, **kwargs):

        def sample(ctx, *, just_say, **kwargs):
            a = ctx.device
            print(f"{just_say} from {a}")
            return f"success+{just_say}"

        return ray.get([ctx.run.remote(fn=sample, **kwargs) for i, ctx in enumerate(self.ctxs)])[0]


world_size = 2
ulysses_degree = 2
pipeline = SimpleWorkerPipeline(world_size=world_size, ulysses_degree=ulysses_degree)

kwargs = dict(just_say="Hello my name is buddy")
result = pipeline(**kwargs)
print(f"This is from global{result}")
