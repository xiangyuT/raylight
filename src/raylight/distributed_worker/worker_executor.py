# This is where the worker is being run, we must use context (ctx)
# Read ray docs for further documentation.
# Manually control torch.distribute using torch.mp.spawn is pain, so ray is the solution
# Based on vLLM and XDit (XFuser) implementation
import ray
from .worker import WorkerContext


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
