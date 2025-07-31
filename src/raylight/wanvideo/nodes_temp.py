from .pipeline import WanClipEncoderFactory
import folder_paths
import torch
import ray
import sys
import raylight


class RayActor:
    def __init__(self):
        self.model = None
        self.run_fn = None

    def load_model(self, load_fn, *args, **kwargs):
        self.model = load_fn(*args, **kwargs)
        return f"Model loaded: {type(self.model)}"

    def set_run_fn(self, run_fn):
        self.run_fn = run_fn
        return "Run function registered."

    def run_model(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if self.run_fn is None:
            raise RuntimeError("Run function not set.")
        return self.run_fn(self.model, *args, **kwargs)


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
        try:
            ray.init(
                ray_cluster_address,
                namespace=ray_cluster_namespace,
                runtime_env={
                    "py_modules": [raylight]
                })
        except Exception as e:
            ray.init(
                runtime_env={
                    "py_modules": [raylight]
                })
            raise RuntimeError(f"Ray connection failed: {e}")

        RemoteActor = ray.remote(RayActor)
        actor = RemoteActor.options(num_gpus=1, name="wanclip-general").remote()
        return (actor,)


class RayWanClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ray_actor": ("RAY_ACTOR",),
                "model_name": (folder_paths.get_filename_list("clip_vision"),),
                "precision": (["float16", "float32", "bfloat16"], {"default": "float16"}),
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
            return clip.get_model(local_rank=0, device_id="0", world_size=1).model

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

