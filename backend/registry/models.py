from django.db import models


class TimeStampedModel(models.Model):
    """
    Abstract base class following Pattern 1 from OOP Extensibility Guide.
    Provides standard timestamp fields for inherited models.
    """
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class ImportedModel(TimeStampedModel):
    """
    Tracks metadata of uploaded models in the database.
    Note: The actual file is strictly stored in MLflow, not locally,
    per user requirements.
    """
    original_filename = models.CharField(max_length=255)
    file_format = models.CharField(max_length=20)
    file_size = models.PositiveBigIntegerField()
    status = models.CharField(max_length=50, default="PENDING")
    
    # MLflow integration fields
    mlflow_name = models.CharField(max_length=255, null=True, blank=True)
    mlflow_version = models.IntegerField(null=True, blank=True)
    run_id = models.CharField(max_length=255, null=True, blank=True)
    model_uri = models.CharField(max_length=500, null=True, blank=True)

    def __str__(self):
        return f"{self.original_filename} ({self.status})"
