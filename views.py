import math
from datetime import date

from django.shortcuts import render
from django.contrib import messages

from .models import Prediction
from .ml import (
    train_model,
    predict_demand,
    cluster_food_items,
    FOOD_ITEMS,
)


def dashboard(request):
    model, columns, metrics = train_model()
    predictions = Prediction.objects.all()[:10]

    return render(request, "forecast/dashboard.html", {
        "total": Prediction.objects.count(),
        "predictions": predictions,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
    })


def predict(request):
    result = None
    today = date.today().isoformat()

    if request.method == "POST":
        try:
            food_item = request.POST.get("food_item", "")
            if food_item not in FOOD_ITEMS:
                raise ValueError("Invalid food item.")

            forecast_date = date.fromisoformat(request.POST.get("forecast_date", ""))
            if forecast_date < date.today():
                raise ValueError("Forecast date aaj ya future ki honi chahiye.")

            temperature = float(request.POST.get("temperature", 25))
            if not 0 <= temperature <= 55:
                raise ValueError("Temperature 0 se 55 °C ke beech honi chahiye.")

            promotion = request.POST.get("promotion") == "1"

            model, columns, _ = train_model()
            quantity = predict_demand(
                model, columns, food_item, forecast_date, temperature, promotion
            )

            Prediction.objects.create(
                food_item=food_item,
                forecast_date=forecast_date,
                predicted_quantity=quantity,
                temperature=temperature,
                promotion=promotion,
            )

            result = {
                "item": food_item,
                "quantity": quantity,
                "stock": math.ceil(quantity * 1.10),
            }

        except (ValueError, TypeError) as exc:
            messages.warning(request, f"Input error: {exc}")

    return render(request, "forecast/predict.html", {
        "items": FOOD_ITEMS,
        "today": today,
        "result": result,
    })


def insights(request):
    return render(request, "forecast/insights.html", {
        "clusters": cluster_food_items(),
    })
