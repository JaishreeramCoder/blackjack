# advisor/serializers.py
from rest_framework import serializers

class BlackjackStateSerializer(serializers.Serializer):
    state = serializers.ListField(
        child=serializers.IntegerField(),
        help_text="List representing blackjack state: [player_sum, usable_ace, dealer_card, cnt_A, cnt_2, ..., cnt_9, cnt_10group]"
    )
