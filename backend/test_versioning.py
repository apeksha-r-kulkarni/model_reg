import requests
import json

url = "http://127.0.0.1:8000/api/models/register/"

def register(filename):
    print(f"=== UPLOADING {filename} ===")
    data = {
        "model_name": "version_test_model",
        "model_type": "Speech",
        "accuracy": "99.0",
        "priority": "1",
        "is_deployable": "on",
        "deployment_points": '["Edge Device"]',
        "versioning": "Auto-increment"
    }
    with open(f"/home/apeksha-ssi021/Shyena/integrated_model/mlflow_registration/test_models/{filename}", "rb") as f:
        res = requests.post(url, data=data, files={"model": f})
        print("Status:", res.status_code)
        print("Response:", res.json())

register("model_v1.pt")
register("model_v2.pt")
register("model_v3.pt")
