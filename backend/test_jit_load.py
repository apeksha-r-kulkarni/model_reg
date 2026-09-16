import torch
import torch.nn as nn

# Create and JIT save
model = nn.Sequential(nn.Linear(10, 2))
scripted = torch.jit.script(model)
torch.jit.save(scripted, "jit.pt")

# Now try to load with torch.load (which is what services.py does)
try:
    loaded = torch.load("jit.pt")
    print("SUCCESS with torch.load:", type(loaded))
except Exception as e:
    print("FAILED with torch.load:", e)
