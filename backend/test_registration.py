import requests

url = "http://127.0.0.1:8000/api/models/register/"

print("=== UPLOADING NEW MODEL WITH METADATA ===")
data = {
    "model_name": "metadata_test_model_onnx",
    "model_type": "Speech",
    "accuracy": "99.0",
    "priority": "1",
    "is_deployable": "on",
    "deployment_points": '["Edge Device"]',
    "versioning": "Auto-increment"
}
try:
    with open("/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models/model_v2.onnx", "rb") as f:
        res = requests.post(url, data=data, files={"model": f})
        print("Status:", res.status_code)
        print("Response:", res.json())
except Exception as e:
    print("Error:", e)
