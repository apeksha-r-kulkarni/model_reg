import torch
import torch.nn as nn
import os

# Create the output directory if it doesn't exist
output_dir = "/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/model/pickles"
os.makedirs(output_dir, exist_ok=True)

# Define a very simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = SimpleModel()

# 1. Save as PyTorch (.pt)
pt_path = os.path.join(output_dir, "simple_model.pt")
torch.save(model, pt_path)
print(f"Created PyTorch model: {pt_path}")

# 2. Save as ONNX (.onnx)
onnx_path = os.path.join(output_dir, "simple_model.onnx")
# Create dummy input for tracing the model
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, onnx_path, 
                  input_names=['input'], output_names=['output'])
print(f"Created ONNX model: {onnx_path}")
