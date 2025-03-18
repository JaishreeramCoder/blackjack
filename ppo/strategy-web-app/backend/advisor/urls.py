# advisor/urls.py
from django.urls import path
from .views import PredictBlackjackAPIView

urlpatterns = [
    path('predict/', PredictBlackjackAPIView.as_view(), name='predict_blackjack'),
]
