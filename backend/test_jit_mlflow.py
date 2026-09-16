import torch
import torch.nn as nn
import mlflow.pytorch

model = nn.Sequential(nn.Linear(10, 2))
scripted = torch.jit.script(model)
torch.jit.save(scripted, "jit.pt")

loaded = torch.load("jit.pt")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.pytorch.log_model(loaded, "jit_model_test")
print("SUCCESS LOGGING JIT TO MLFLOW")
