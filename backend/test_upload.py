import requests

url = "http://localhost:8000/api/models/register/"
data = {
    "model_name": "test_onnx_upload",
    "model_type": "Classification",
    "accuracy": "95.5",
    "deployment_points": "Cloud App",
    "versioning": "Auto-increment"
}
files = {
    "model": open("/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models/model_v1.onnx", "rb")
}

try:
    response = requests.post(url, data=data, files=files)
    print("Status Code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Request failed:", e)
