# advisor/urls.py
from django.urls import path
from .views import PredictBlackjackAPIView
from .views import health
urlpatterns = [
    path('predict/', PredictBlackjackAPIView.as_view(), name='predict_blackjack'),
    path('health/', health, name='health'),
]
