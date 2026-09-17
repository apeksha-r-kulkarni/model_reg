import os
import torch
import requests

url = "http://127.0.0.1:8000/api/models/register/"

print("=== UPLOADING NEW MODEL WITH METADATA ===")
data = {
    "model_name": "metadata_test_model_pt",
    "model_type": "Vision",
    "accuracy": "99.0",
    "priority": "1",
    "is_deployable": "on",
    "deployment_points": '["Edge Device"]',
    "versioning": "Auto-increment"
}

dummy_path = "test_dummy.pt"
try:
    # Generate a tiny dummy PyTorch model on the fly
    model = torch.nn.Linear(10, 2)
    torch.save(model, dummy_path)

    with open(dummy_path, "rb") as f:
        res = requests.post(url, data=data, files={"model": f})
        print("Status:", res.status_code)
        print("Response:", res.json())
except Exception as e:
    print("Error:", e)
finally:
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
