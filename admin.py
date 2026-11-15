from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ("food_item", "forecast_date", "predicted_quantity",
                    "temperature", "promotion", "created_at")
    list_filter = ("food_item", "promotion")
