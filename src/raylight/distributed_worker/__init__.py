import torch

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

