import os
import django

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from registry.models import ImportedModel

import requests
import mlflow
import torch
import time

DJANGO_URL = "http://127.0.0.1:8000/api/models/"
REGISTER_URL = "http://127.0.0.1:8000/api/models/register/"
MLFLOW_URL = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URL)
client = mlflow.MlflowClient()

def create_model(name):
    data = {
        "model_name": name,
        "model_type": "Vision",
        "accuracy": "99.0",
        "priority": "1",
        "is_deployable": "on",
        "deployment_points": '["Edge Device"]',
        "versioning": "Auto-increment"
    }
    dummy_path = f"{name}.pt"
    model = torch.nn.Linear(10, 2)
    torch.save(model, dummy_path)
    try:
        with open(dummy_path, "rb") as f:
            res = requests.post(REGISTER_URL, data=data, files={"model": f})
            res.raise_for_status()
            return res.json()
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

def get_django_models():
    res = requests.get(DJANGO_URL)
    res.raise_for_status()
    return {m["name"]: m for m in res.json().get("models", [])}

print("=== 1. Registering test model ===")
res_data = create_model("test_sync_model")
print("Registered:", res_data)

print("\n=== 2. Verify SQLite status is REGISTERED ===")
db_record_id = res_data["db_record_id"]
record = ImportedModel.objects.get(id=db_record_id)
print(f"Status in SQLite for {db_record_id}: {record.status}")
assert record.status == "REGISTERED"

print("\n=== 3. Deleting model from MLflow ===")
client.delete_registered_model(name="test_sync_model")
print("Deleted test_sync_model from MLflow.")

print("\n=== 4. Refreshing GET /api/models/ ===")
django_models = get_django_models()
if "test_sync_model" not in django_models:
    print("SUCCESS: test_sync_model does not appear in Django API.")
else:
    print("FAIL: test_sync_model still visible!")

print("\n=== 5. Verify SQLite record becomes DELETED ===")
record.refresh_from_db()
print(f"Status in SQLite for {db_record_id}: {record.status}")
assert record.status == "DELETED"

print("\n=== 6. Verify normal registration still works ===")
res2 = create_model("test_sync_model_2")
print("Successfully registered second model:", res2["model_name"])

# Cleanup
client.delete_registered_model(name="test_sync_model_2")

print("\n=== All Tests Passed ===")
