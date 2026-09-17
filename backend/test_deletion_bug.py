import requests
import mlflow
import os
import torch
import time

DJANGO_URL = "http://127.0.0.1:8000/api/models/"
REGISTER_URL = "http://127.0.0.1:8000/api/models/register/"
MLFLOW_URL = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URL)
client = mlflow.MlflowClient()

def create_model(name):
    # Upload a dummy model to create it in the system
    print(f"--- Creating {name} ---")
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
            print("Successfully registered", name)
            return res.json()
    finally:
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

def get_django_models():
    res = requests.get(DJANGO_URL)
    res.raise_for_status()
    return {m["name"]: m for m in res.json().get("models", [])}

print("=== 1. Test registration of models ===")
m1 = create_model("test_active_model")
m2 = create_model("test_empty_model")
m3 = create_model("test_deleted_model")

# Wait for MLflow index
time.sleep(2)

print("\n=== 2. Verify all exist ===")
django_models = get_django_models()
assert "test_active_model" in django_models, "Active model missing!"
assert "test_empty_model" in django_models, "Empty model missing!"
assert "test_deleted_model" in django_models, "Deleted model missing!"
print("All 3 models appear correctly.")

print("\n=== 3. Delete versions from test_empty_model ===")
versions = client.search_model_versions("name='test_empty_model'")
for v in versions:
    client.delete_model_version(name="test_empty_model", version=v.version)
print("Deleted versions for test_empty_model.")

print("\n=== 4. Delete entirely test_deleted_model ===")
client.delete_registered_model(name="test_deleted_model")
print("Deleted registered model test_deleted_model.")

print("\n=== 5. Verify Django API Results ===")
django_models = get_django_models()

print("Checking test_active_model...")
if "test_active_model" in django_models:
    print("SUCCESS: test_active_model is present.")
else:
    print("FAIL: test_active_model is missing!")

print("Checking test_empty_model...")
if "test_empty_model" not in django_models:
    print("SUCCESS: test_empty_model is correctly HIDDEN (versions deleted).")
else:
    print("FAIL: test_empty_model still visible!")

print("Checking test_deleted_model...")
if "test_deleted_model" not in django_models:
    print("SUCCESS: test_deleted_model is correctly HIDDEN (model deleted).")
else:
    print("FAIL: test_deleted_model still visible!")

# Cleanup
client.delete_registered_model(name="test_active_model")
client.delete_registered_model(name="test_empty_model")
print("\n=== Cleanup complete ===")
