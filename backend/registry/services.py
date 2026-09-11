import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

from .models import ImportedModel

# MLflow tracking server — must be running at this address before registering models.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


class ModelRegistrationError(Exception):
    """Custom exception raised when model registration fails."""
    pass


def register_mlflow_model(file_path: str, model_name: str, original_filename: str, file_size: int) -> dict:
    """
    Business logic to load a joblib model and register it with MLflow.
    Tracks metadata in the Django DB, but does NOT store the file locally.
    Raises specific exceptions on failure.
    """
    
    # Create the metadata DB record first (status: PENDING)
    # The actual file isn't stored in Django; only the metadata.
    file_format = original_filename.split(".")[-1].lower() if "." in original_filename else "unknown"
    db_record = ImportedModel.objects.create(
        original_filename=original_filename,
        file_format=file_format,
        file_size=file_size,
        status="PENDING",
        mlflow_name=model_name
    )

    # ── 1. Load the model via joblib ──────────────────────────────────────
    try:
        model = joblib.load(file_path)
    except (OSError, ValueError, EOFError, AttributeError, ImportError) as load_err:
        db_record.status = "FAILED_LOADING"
        db_record.save()
        raise ModelRegistrationError(f"Could not load model file: {load_err}") from load_err

    # ── 2. Connect to MLflow ──────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ── 3. Log model in a run, then register it ───────────────────────────
    try:
        with mlflow.start_run() as run:
            # MLflow 3.x: sk_model is the first positional arg, name is keyword
            mlflow.sklearn.log_model(model, name="model")
            run_id = run.info.run_id

        # Construct the artifact URI after the run is finalised
        model_uri = f"runs:/{run_id}/model"

        # Register the logged model in the MLflow Model Registry
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

    except MlflowException as mlflow_err:
        db_record.status = "FAILED_MLFLOW"
        db_record.save()
        raise ModelRegistrationError(f"MLflow registration failed: {mlflow_err}") from mlflow_err

    # ── 4. Update the DB record and return success data ───────────────────
    db_record.status = "REGISTERED"
    db_record.mlflow_version = int(result.version)
    db_record.run_id = run_id
    db_record.model_uri = model_uri
    db_record.save()

    return {
        "model_name": result.name,
        "version": int(result.version),
        "run_id": run_id,
        "model_uri": model_uri,
        "db_record_id": db_record.id,
    }


def list_registered_models() -> list[dict]:
    """
    Query the MLflow Model Registry and return all registered models
    with ALL their versions (not just latest per stage).
    Raises ModelRegistrationError on connection failure.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = mlflow.MlflowClient()
        registered_models = client.search_registered_models()
    except MlflowException as err:
        raise ModelRegistrationError(f"Could not connect to MLflow: {err}") from err

    results = []
    for model in registered_models:
        # Fetch every version for this model, sorted newest first
        try:
            all_versions = client.search_model_versions(f"name='{model.name}'")
        except MlflowException as err:
            raise ModelRegistrationError(
                f"Could not fetch versions for model '{model.name}': {err}"
            ) from err

        versions_data = sorted(
            [
                {
                    "version": int(v.version),
                    "status": v.status,
                    "run_id": v.run_id,
                    "model_uri": f"runs:/{v.run_id}/model",
                    "creation_timestamp": v.creation_timestamp,
                }
                for v in all_versions
            ],
            key=lambda x: x["version"],
            reverse=True,  # newest version first
        )

        results.append({
            "name": model.name,
            "all_versions": versions_data,
            "creation_timestamp": model.creation_timestamp,
            "last_updated_timestamp": model.last_updated_timestamp,
        })

    return results
