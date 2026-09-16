import torch
import torch.nn as nn
import os
import mlflow.pytorch

model = nn.Sequential(nn.Linear(10, 2))
scripted_model = torch.jit.script(model)

torch.save(scripted_model, "/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models/model_v1_jit.pt")
