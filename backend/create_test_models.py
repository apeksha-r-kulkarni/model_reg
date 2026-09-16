import os
import torch
import torch.nn as nn
import onnx
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

output_dir = "/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models"
os.makedirs(output_dir, exist_ok=True)

# 1. Define four different PyTorch portable model architectures using standard layers
model_v1 = nn.Sequential(nn.Linear(10, 2))
model_v2 = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))

# V3: CNN
model_v3 = nn.Sequential(
    nn.Conv2d(1, 4, 3),
    nn.ReLU(),
    nn.Flatten(1),
    nn.Linear(4 * 26 * 26, 10)
)

# V4: RNN
# Note: LSTM returns a tuple, which nn.Sequential can't handle directly,
# but we can use TorchScript to make a portable LSTM wrapper if needed.
# Since we just need portable models, let's use another standard architecture for V4
# to keep it fully portable without custom classes.
model_v4 = nn.Sequential(
    nn.Linear(50, 25),
    nn.Sigmoid(),
    nn.Linear(25, 5)
)

models = {
    "model_v1": (model_v1, torch.randn(1, 10)),
    "model_v2": (model_v2, torch.randn(1, 20)),
    "model_v3": (model_v3, torch.randn(1, 1, 28, 28)),
    "model_v4": (model_v4, torch.randn(1, 50)),
}

print("="*80)
print("GENERATING & VALIDATING MODELS")
print("="*80)

results = []

for name, (model, dummy_input) in models.items():
    pt_path = os.path.join(output_dir, f"{name}.pt")
    onnx_path = os.path.join(output_dir, f"{name}.onnx")
    
    # --- PYTORCH ---
    torch.save(model, pt_path)
    
    try:
        # EXACT validation used by current application (services.py)
        loaded_pt = torch.load(pt_path, map_location='cpu')
        
        # Also check mlflow flavor compatibility
        import mlflow.pytorch
        # We don't actually start a run, we just check if it's a Module
        if not isinstance(loaded_pt, nn.Module):
            raise TypeError("Not an nn.Module")
            
        pt_status = "VALID"
        pt_details = f"Arch: {loaded_pt.__class__.__name__}, Params: {sum(p.numel() for p in loaded_pt.parameters())}"
    except Exception as e:
        pt_status = f"INVALID"
        pt_details = str(e).split('\n')[0]
        
    results.append((f"{name}.pt", "PyTorch", pt_status, pt_details))
    
    # --- ONNX ---
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      opset_version=14)
    
    try:
        # EXACT validation used by current application (services.py)
        loaded_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(loaded_onnx)
        
        onnx_status = "VALID"
        graph = loaded_onnx.graph
        onnx_details = f"Nodes: {len(graph.node)}, Inputs: {len(graph.input)}, Outputs: {len(graph.output)}"
    except Exception as e:
        onnx_status = f"INVALID"
        onnx_details = str(e).split('\n')[0]
        
    results.append((f"{name}.onnx", "ONNX", onnx_status, onnx_details))

# Print Report
print(f"{'FILENAME':<15} | {'FORMAT':<10} | {'STATUS':<8} | {'DETAILS'}")
print("-" * 80)
for res in results:
    print(f"{res[0]:<15} | {res[1]:<10} | {res[2]:<8} | {res[3]}")
print("="*80)
print(f"All models saved in: {output_dir}")
