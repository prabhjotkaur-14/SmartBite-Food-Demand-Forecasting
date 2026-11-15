from django.db import models


class Prediction(models.Model):

    food_item = models.CharField(max_length=100)
    forecast_date = models.DateField()
    predicted_quantity = models.PositiveIntegerField()
    temperature = models.FloatField(default=25)
    promotion = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.food_item
