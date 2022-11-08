import numpy as np
import torch

# FP32

model = torch.load("./data/clip.bin")
np.savez("./data/clip.npz", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
model = torch.load("./data/vae_v1_5.bin")
np.savez("./data/vae_v1_5.npz", **{k: v.numpy() for k, v in model.items()})
model = torch.load("./data/unet_v1_5.bin")
np.savez("./data/unet_v1_5.npz", **{k: v.numpy() for k, v in model.items()})

# FP16

model = torch.load("./data/clip_v1_5_fp16.bin")
np.savez("./data/clip_v1_5_fp16.npz", **{k: v.numpy() for k, v in model.items() if "text_model" in k})
model = torch.load("./data/vae_v1_5_fp16.bin")
np.savez("./data/vae_v1_5_fp16.npz", **{k: v.numpy() for k, v in model.items()})
model = torch.load("./data/unet_v1_5_fp16.bin")
np.savez("./data/unet_v1_5_fp16.npz", **{k: v.numpy() for k, v in model.items()})