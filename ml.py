from pathlib import Path
from datetime import date as date_type

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FOOD_ITEMS = [
    "Veg Burger",
    "Paneer Wrap",
    "Pizza",
    "Pasta",
    "Cold Coffee",
]

BASE_DEMAND = {
    "Veg Burger": 35,
    "Paneer Wrap": 25,
    "Pizza": 30,
    "Pasta": 20,
    "Cold Coffee": 28,
}

MODEL_PATH = Path(__file__).resolve().parent / "demand_model.joblib"

_cache = {}


def create_dataset():
    """Synthetic demo data (real restaurant sales nahi hai)."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2025-01-01")
    rows = []

    for item in FOOD_ITEMS:
        base = BASE_DEMAND[item]
        for day in range(365):
            d = start + pd.Timedelta(days=day)
            is_weekend = 1 if d.dayofweek >= 5 else 0
            temperature = 25 + 10 * np.sin(2 * np.pi * (d.dayofyear - 80) / 365) \
                + rng.normal(0, 2)
            promotion = int(rng.random() < 0.2)

            temp_effect = 0.8 if item == "Cold Coffee" else 0.1
            qty = (base
                   + is_weekend * 8
                   + promotion * 10
                   + (temperature - 25) * temp_effect
                   + rng.normal(0, 3))

            rows.append({
                "date": d,
                "food_item": item,
                "temperature": round(float(temperature), 1),
                "promotion": promotion,
                "is_weekend": is_weekend,
                "day_of_week": d.dayofweek,
                "month": d.month,
                "quantity": max(0, int(round(qty))),
            })

    return pd.DataFrame(rows)


def _features(df):
    X = df[["temperature", "promotion", "is_weekend", "day_of_week", "month"]].copy()
    for item in FOOD_ITEMS:
        X[f"item_{item}"] = (df["food_item"] == item).astype(int)
    return X


def train_model(force=False):
    """Returns (model, columns, metrics). Chronological 80/20 split."""
    if not force and "model" in _cache:
        return _cache["model"], _cache["columns"], _cache["metrics"]

    if not force and MODEL_PATH.exists():
        try:
            saved = joblib.load(MODEL_PATH)
            _cache.update(saved)
            return saved["model"], saved["columns"], saved["metrics"]
        except Exception:
            pass  # corrupt/old file -> retrain

    df = create_dataset().sort_values("date").reset_index(drop=True)
    cutoff = df["date"].quantile(0.8)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]

    X_train, y_train = _features(train), train["quantity"]
    X_test, y_test = _features(test), test["quantity"]

    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        "r2": round(float(r2_score(y_test, pred)), 3),
    }
    columns = list(X_train.columns)

    bundle = {"model": model, "columns": columns, "metrics": metrics}
    joblib.dump(bundle, MODEL_PATH)
    _cache.update(bundle)
    return model, columns, metrics


def predict_demand(model, columns, food_item, forecast_date, temperature, promotion):
    if isinstance(forecast_date, str):
        forecast_date = date_type.fromisoformat(forecast_date)

    row = {c: 0 for c in columns}
    row["temperature"] = float(temperature)
    row["promotion"] = int(promotion)
    row["is_weekend"] = 1 if forecast_date.weekday() >= 5 else 0
    row["day_of_week"] = forecast_date.weekday()
    row["month"] = forecast_date.month
    key = f"item_{food_item}"
    if key in row:
        row[key] = 1

    X = pd.DataFrame([row], columns=columns)
    return max(0, int(round(model.predict(X)[0])))


def cluster_food_items(n_clusters=3):
    df = create_dataset()
    stats = df.groupby("food_item")["quantity"].agg(["mean", "std"]).reset_index()

    scaled = StandardScaler().fit_transform(stats[["mean", "std"]])
    k = min(n_clusters, len(stats))
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
    stats["cluster"] = labels + 1

    return stats.to_dict("records")
