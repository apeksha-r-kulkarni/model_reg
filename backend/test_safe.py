import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 2))
torch.save(model, "safe_model.pt")

try:
    loaded = torch.load("safe_model.pt", map_location="cpu")
    print("LOAD SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
