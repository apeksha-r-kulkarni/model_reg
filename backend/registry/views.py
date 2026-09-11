import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods


# MLflow tracking server — must be running at this address before registering models.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"


def register_page(request):
    """Render the model registration page."""
    return render(request, "register.html")


@csrf_exempt
@require_http_methods(["POST"])
def register_model(request):
    """
    POST /api/models/register/

    Accepts multipart/form-data with:
        model_name  (str)  — name to register the model under in MLflow
        model       (file) — a .pkl file produced by joblib/sklearn

    Returns JSON:
        {
            "success":    true,
            "message":    "Model registered successfully.",
            "model_name": "<name>",
            "version":    <int>,
            "run_id":     "<uuid>",
            "model_uri":  "runs:/<run_id>/model"
        }

    ⚠️  SECURITY NOTE:
    Joblib/pickle files can execute arbitrary Python code when deserialized.
    Only accept .pkl files from trusted sources. This endpoint must NOT be
    exposed to the public internet without additional authentication.
    """

    # ── 1. Input validation (before any file I/O) ────────────────────────────

    model_file = request.FILES.get("model")
    model_name = request.POST.get("model_name", "").strip()

    if not model_file:
        return JsonResponse(
            {"success": False, "error": "No model file uploaded."},
            status=400,
        )

    if not model_name:
        return JsonResponse(
            {"success": False, "error": "Model name is required."},
            status=400,
        )

    if not model_file.name.lower().endswith(".pkl"):
        return JsonResponse(
            {"success": False, "error": "Only .pkl files are supported."},
            status=400,
        )

    # ── 2. Write upload to a secure temp file (not the project root) ─────────

    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            for chunk in model_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        # ── 3. Load the model via joblib ──────────────────────────────────────

        try:
            model = joblib.load(tmp_path)
        except Exception as load_err:
            return JsonResponse(
                {"success": False, "error": f"Could not load model file: {load_err}"},
                status=422,
            )

        # ── 4. Connect to MLflow ──────────────────────────────────────────────

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # ── 5. Log model in a run, then register it ───────────────────────────

        try:
            with mlflow.start_run() as run:
                # MLflow 3.x: sk_model is the first positional arg, name is keyword
                mlflow.sklearn.log_model(model, name="model")
                run_id = run.info.run_id

            # Construct the artifact URI after the run is finalised
            model_uri = f"runs:/{run_id}/model"

            # Register the logged model in the MLflow Model Registry
            result = mlflow.register_model(model_uri=model_uri, name=model_name)

        except Exception as mlflow_err:
            return JsonResponse(
                {
                    "success": False,
                    "error": f"MLflow registration failed: {mlflow_err}",
                },
                status=502,
            )

        # ── 6. Return success ─────────────────────────────────────────────────

        return JsonResponse(
            {
                "success": True,
                "message": "Model registered successfully.",
                "model_name": result.name,
                "version": int(result.version),
                "run_id": run_id,
                "model_uri": model_uri,
            }
        )

    finally:
        # Always remove the temp file, regardless of success or error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

