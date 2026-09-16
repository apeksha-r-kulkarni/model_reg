import torch
import torch.nn as nn
import mlflow.pytorch

model = nn.Sequential(nn.Linear(10, 2))
state_dict = model.state_dict()

# simulate what services.py does
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.pytorch.log_model(state_dict, artifact_path="model_state_dict")
print("Successfully logged state_dict!")
