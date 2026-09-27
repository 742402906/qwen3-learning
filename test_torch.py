import torch

print("PyTorch版本:")
print(torch.__version__)

print("MPS是否可用:")
print(torch.backends.mps.is_available())
