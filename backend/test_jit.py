import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2

model = torch.jit.script(MyModel())
torch.jit.save(model, "jit_model.pt")

try:
    # This is exactly what services.py does
    loaded = torch.load("jit_model.pt", map_location="cpu")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
