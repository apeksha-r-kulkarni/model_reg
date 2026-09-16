import torch
import torch.nn as nn
import mlflow.pytorch

class MyModel(nn.Module):
    def forward(self, x):
        return x * 2

model = MyModel()
torch.save(model.state_dict(), "state_dict.pt")

try:
    loaded = torch.load("state_dict.pt", map_location="cpu")
    print("LOAD SUCCESS")
    with mlflow.start_run():
        mlflow.pytorch.log_model(loaded, artifact_path="model")
    print("MLFLOW SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
