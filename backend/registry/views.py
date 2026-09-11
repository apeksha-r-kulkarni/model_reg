import os
import tempfile

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .services import register_mlflow_model, list_registered_models, ModelRegistrationError


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

    Returns JSON on success or failure.

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

        # ── 3. Call the business logic service ────────────────────────────────

        try:
            result_data = register_mlflow_model(
                file_path=tmp_path,
                model_name=model_name,
                original_filename=model_file.name,
                file_size=model_file.size
            )
        except ModelRegistrationError as err:
            return JsonResponse(
                {"success": False, "error": str(err)},
                status=422 if "Could not load" in str(err) else 502,
            )

        # ── 4. Return success ─────────────────────────────────────────────────

        return JsonResponse(
            {
                "success": True,
                "message": "Model registered successfully.",
                **result_data,
            }
        )

    finally:
        # Always remove the temp file, regardless of success or error
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@require_http_methods(["GET"])
def list_models(request):
    """
    GET /api/models/

    Returns a JSON list of all models registered in the MLflow Model Registry.
    """
    try:
        models = list_registered_models()
    except ModelRegistrationError as err:
        return JsonResponse(
            {"success": False, "error": str(err)},
            status=502,
        )

    return JsonResponse({"success": True, "models": models})
