from .pipeline import WanClipEncoderFactory
import folder_paths
import torch
import ray
import sys
import os
import types

def print_custom_node_modules(node_folder_name="raylight"):
    print(f"\n--- Loaded modules under '{node_folder_name}' ---")

    for name, module in sys.modules.items():
        if name.startswith(node_folder_name):
            print(f"{name} → {getattr(module, '__file__', 'built-in or dynamically created')}")

# --- Ensure 'raylight' module is importable by Ray ---
raylight_path = os.path.abspath(os.path.join(__file__, "../.."))
if raylight_path not in sys.path:
    sys.path.insert(0, raylight_path)

# --- Import actor ---
from ..distributed_worker import RayActor

# --- Patch sys.modules so Ray workers can import this actor ---
module_name = "raylight.distributed_worker"
if module_name not in sys.modules:
    # Register a fake module and attach the class
    fake_module = types.ModuleType(module_name)
    fake_module.RayActor = RayActor
    sys.modules[module_name] = fake_module

class GeneralRayInitializer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_cluster_address": ("STRING", {"default": "local"}),
                "ray_cluster_namespace": ("STRING", {"default": "default"}),
            }
        }

    RETURN_TYPES = ("RAY_ACTOR",)
    RETURN_NAMES = ("ray_actor",)
    FUNCTION = "spawn_actor"
    CATEGORY = "Raylight"

    def spawn_actor(self, ray_cluster_address, ray_cluster_namespace):
        print_custom_node_modules("raylight")
        try:
            ray.init(address=ray_cluster_address, namespace=ray_cluster_namespace)
        except Exception as e:
            ray.init(namespace=ray_cluster_namespace)
            raise RuntimeError(f"Ray connection failed: {e}")

        RemoteActor = ray.remote(RayActor)
        actor = RemoteActor.options(name="wanclip-general").remote()
        return (actor,)


class RayWanClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actor": ("RAY_ACTOR",),
                "model_name": (folder_paths.get_filename_list("clip_vision"),),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
            }
        }

    RETURN_TYPES = ("RAY_ACTOR",)
    RETURN_NAMES = ("loaded_actor",)
    FUNCTION = "load_clip_model"
    CATEGORY = "Raylight"

    def load_clip_model(self, ray_actor, model_name, precision):
        model_path = folder_paths.get_full_path("clip_vision", model_name)

        def clip_loader(path, precision):
            clip = WanClipEncoderFactory(
                model_path=path,
                dtype=getattr(torch, precision),
                model_dtype=getattr(torch, precision)
            )
            return clip.get_model(local_rank=0, device_id="cuda:0", world_size=1).model

        ray.get(ray_actor.load_model.remote(clip_loader, model_path, precision))

        def clip_runner(model, input_tensor):
            return model(input_tensor)

        ray.get(ray_actor.set_run_fn.remote(clip_runner))

        return (ray_actor,)


NODE_CLASS_MAPPINGS = {
    "GeneralRayInitializer": GeneralRayInitializer,
    "RayWanClipLoader": RayWanClipLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeneralRayInitializer": "Ray Init Actor (General)",
    "RayWanClipLoader": "Ray Load WanClip",
}

