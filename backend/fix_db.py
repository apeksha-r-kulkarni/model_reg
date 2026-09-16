import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from registry.models import ImportedModel

# Clear architecture string to avoid Foreign Key constraint error during migration
for m in ImportedModel.objects.all():
    m.architecture = None
    m.save()
print("Cleared existing architecture fields.")
