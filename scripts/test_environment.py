import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    print("CUDA version (torch):", torch.version.cuda)
else:
    print("No GPU detected — check driver/torch install.")
