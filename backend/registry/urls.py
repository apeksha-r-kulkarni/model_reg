from django.urls import path

from . import views

urlpatterns = [
    # GET  http://127.0.0.1:8000/
    path("", views.register_page, name="register_page"),

    # POST http://127.0.0.1:8000/api/models/register/
    path("api/models/register/", views.register_model, name="register_model"),
]
