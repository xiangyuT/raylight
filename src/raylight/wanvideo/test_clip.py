from .pipeline import WanClipEncoderFactory
from PIL import Image
from torchvision import transforms
from comfy.clip_vision import clip_preprocess, ClipVisionModel
import torch

model_path = "/home/kxn/ComfyShard/ComfyUI/models/clip_vision/clip_xlm.safetensors"
clip_vision = WanClipEncoderFactory(dtype=torch.bfloat16, model_path=model_path, model_dtype=torch.bfloat16)

image_mean = [0.48145466, 0.4578275, 0.40821073]
image_std = [0.26862954, 0.26130258, 0.27577711]

img = Image.open("test1.png").convert("RGB")  # force RGB, 3 channels

to_tensor = transforms.ToTensor()
img = to_tensor(img)
image = img.unsqueeze(0)

negative_clip_embeds = None
device = torch.device("cpu")

if isinstance(clip_vision, ClipVisionModel):
    clip_embeds = clip_vision.encode_image(image).penultimate_hidden_states.to(device)
    print("success")
else:
    pixel_values = clip_preprocess(image.to(device), size=224, mean=image_mean, std=image_std).float()
    clip_embeds = clip_vision.visual(pixel_values)

