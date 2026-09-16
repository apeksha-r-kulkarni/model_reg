import requests
import sys

url = "http://127.0.0.1:8000/api/models/register/"

print("=== TESTING PT MODEL ===")
data_pt = {
    "model_name": "test_pt_final",
    "model_type": "Classification",
    "accuracy": "89.0",
    "deployment_points": '["Edge Device"]',
    "versioning": "Auto-increment"
}
try:
    with open("/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models/model_v1.pt", "rb") as f:
        files_pt = {"model": f}
        res = requests.post(url, data=data_pt, files=files_pt)
        print("Status:", res.status_code)
        print("Response:", res.json())
except Exception as e:
    print("Error:", e)

print("\n=== TESTING ONNX MODEL ===")
data_onnx = {
    "model_name": "test_onnx_final",
    "model_type": "Classification",
    "accuracy": "92.5",
    "deployment_points": '["Cloud App"]',
    "versioning": "Auto-increment"
}
try:
    with open("/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models/model_v1.onnx", "rb") as f:
        files_onnx = {"model": f}
        res = requests.post(url, data=data_onnx, files=files_onnx)
        print("Status:", res.status_code)
        print("Response:", res.json())
except Exception as e:
    print("Error:", e)
