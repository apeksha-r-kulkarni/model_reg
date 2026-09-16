import torch
import torch.nn as nn
import os
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

model = nn.Sequential(nn.Linear(10, 2))
torch.save(model, "safe_model.pt")

try:
    loaded = torch.load("safe_model.pt", map_location="cpu")
    print("LOAD SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
