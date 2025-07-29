import torch
import ray


class RayEngine:
    def __init__(self, name):
        self.name = name
        self.cuTensor = torch.randn(4)

    def run(self):
        return [self.name, self.cuTensor * 3]


ray.init()
RemoteRay = ray.remote(RayEngine)
worker1 = RemoteRay.options(name="gpu-1").remote("cu1")
worker2 = RemoteRay.options(name="gpu-2").remote("cu2")

print(ray.get(ray.get_actor("gpu-1").run.remote()))
print(ray.get(ray.get_actor("gpu-2").run.remote()))
