import json
import numpy as np
from django.http import JsonResponse, HttpResponseBadRequest
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import BlackjackStateSerializer

# ----------------------------
# Environment Definition
# ----------------------------
class BlackjackEnvCountingFirstMove(gym.Env):
    """
    Custom Blackjack environment with card counting and a first-move flag.

    State (MultiDiscrete):
      [player_sum, usable_ace, dealer_card, is_first_move, cnt_A, cnt_2, ..., cnt_9, cnt_10group]

      - player_sum: Sum of player's hand (0 to 31)
      - usable_ace: 0 (no usable ace) or 1 (usable ace exists)
      - dealer_card: Dealer's face-up card (1 for Ace, 2–10 for number cards)
      - is_first_move: 1 if it's the first decision of the episode, 0 otherwise.
      - cnt_A: Count for Ace (0–4)
      - cnt_2 to cnt_9: Count for cards 2 through 9 (each 0–4)
      - cnt_10group: Count for 10, Jack, Queen, King (0–16)

    Actions (Discrete(4)):
      0: HIT      – Request another card.
      1: STK      – Stand.
      2: DBL      – Double Down (allowed only on the first move).
      3: SUR      – Surrender (allowed only on the first move).

    On moves after the first, only HIT (0) and STK (1) are allowed.
    
    Reward Structure:
      - Blackjack pays 3:2 (payout_blackjack=1.5) if only the player has blackjack.
      - Regular win pays 1:1.
      - Push returns 0.
      - Loss costs the bet.
      - Surrender returns -0.5 times the base bet.
      - Double Down outcomes are scaled (bet multiplied by 2).
    """
    def __init__(self, payout_blackjack=1.5, deck_threshold=15):
        super(BlackjackEnvCountingFirstMove, self).__init__()
        # Define the action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        # Observation space:
        # [player_sum (32), usable_ace (2), dealer_card (11), is_first_move (2),
        #  cnt_A (5), cnt_2,...,cnt_9 (each 5), cnt_10group (17)]
        self.observation_space = spaces.MultiDiscrete([32, 2, 11, 2] + [5]*9 + [17])
        
        self.payout_blackjack = payout_blackjack
        self.base_bet = 1.0
        self.deck_threshold = deck_threshold
        
        self._init_deck()
        self.reset()
        
    def _init_deck(self):
        """Initialize a single deck and reset card counts."""
        self.deck = []
        self.deck += ['A'] * 4
        for card in range(2, 10):
            self.deck += [str(card)] * 4
        self.deck += ['10'] * 16
        random.shuffle(self.deck)
        
        self.card_counts = {'A': 0}
        for card in range(2, 10):
            self.card_counts[str(card)] = 0
        self.card_counts['10'] = 0
        
    def _draw_card(self):
        if len(self.deck) == 0:
            self._init_deck()
        card = self.deck.pop()
        if card == 'A':
            self.card_counts['A'] = min(self.card_counts['A'] + 1, 4)
        elif card == '10':
            self.card_counts['10'] = min(self.card_counts['10'] + 1, 16)
        else:
            self.card_counts[card] = min(self.card_counts[card] + 1, 4)
        return card
    
    def _hand_value(self, hand):
        total = 0
        ace_count = 0
        for card in hand:
            if card == 'A':
                total += 1
                ace_count += 1
            else:
                total += int(card)
        usable_ace = 0
        if ace_count > 0 and total + 10 <= 21:
            total += 10
            usable_ace = 1
        return total, usable_ace
    
    def _card_value(self, card):
        return 1 if card == 'A' else int(card)
    
    def _get_observation(self):
        player_sum, usable_ace = self._hand_value(self.player_hand)
        dealer_card_val = self._card_value(self.dealer_hand[0])
        first_move_flag = 1 if self.first_move else 0
        counts = [self.card_counts['A']]
        for card in range(2, 10):
            counts.append(self.card_counts[str(card)])
        counts.append(self.card_counts['10'])
        obs = np.array([player_sum, usable_ace, dealer_card_val, first_move_flag] + counts, dtype=np.int32)
        return obs
    
    def reset(self, seed=None, options=None):
        self.first_move = True
        self.done = False
        self.natural_blackjack = False
        if len(self.deck) < self.deck_threshold:
            self._init_deck()
        self.player_hand = [self._draw_card(), self._draw_card()]
        self.dealer_hand = [self._draw_card(), self._draw_card()]
        player_total, _ = self._hand_value(self.player_hand)
        dealer_total, _ = self._hand_value(self.dealer_hand)
        if player_total == 21:
            self.reward = 0.0 if dealer_total == 21 else self.payout_blackjack * self.base_bet
            self.natural_blackjack = True
        else:
            self.reward = 0.0
        return self._get_observation(), {}
    
    def step(self, action):
        if self.natural_blackjack:
            self.natural_blackjack = False
            self.done = True
            info = {"bet": 1.0}
            return self._get_observation(), self.reward, True, False, info

        if self.done:
            return self._get_observation(), 0.0, True, False, {}
        
        if not self.first_move and action in [2, 3]:
            self.done = True
            return self._get_observation(), -1.0, True, False, {"illegal_action": True}
        
        if action == 0:  # HIT
            card = self._draw_card()
            self.player_hand.append(card)
            player_total, _ = self._hand_value(self.player_hand)
            if player_total > 21:
                self.done = True
                reward = -self.base_bet
            else:
                reward = 0.0
            self.first_move = False
            return self._get_observation(), reward, self.done, False, {}
        
        elif action == 1:  # STAND
            reward = self._dealer_play()
            self.done = True
            return self._get_observation(), reward, self.done, False, {}
        
        elif action == 2:  # DOUBLE DOWN
            self.first_move = False
            card = self._draw_card()
            self.player_hand.append(card)
            player_total, _ = self._hand_value(self.player_hand)
            if player_total > 21:
                reward = -2 * self.base_bet
                self.done = True
                return self._get_observation(), reward, self.done, False, {}
            reward = self._dealer_play(double_down=True)
            self.done = True
            return self._get_observation(), reward, self.done, False, {}
        
        elif action == 3:  # SURRENDER
            self.first_move = False
            self.done = True
            reward = -0.5 * self.base_bet
            return self._get_observation(), reward, self.done, False, {}
        else:
            self.done = True
            return self._get_observation(), -1.0, True, False, {"illegal_action": True}
    
    def _dealer_play(self, double_down=False):
        player_total, _ = self._hand_value(self.player_hand)
        dealer_total, _ = self._hand_value(self.dealer_hand)
        while dealer_total < 17:
            card = self._draw_card()
            self.dealer_hand.append(card)
            dealer_total, _ = self._hand_value(self.dealer_hand)
        bet = self.base_bet * (2 if double_down else 1)
        if dealer_total > 21:
            return bet
        elif dealer_total > player_total:
            return -bet
        elif dealer_total < player_total:
            return bet
        else:
            return 0.0
       
    def render(self, mode='human'):
        player_total, usable = self._hand_value(self.player_hand)
        dealer_total, _ = self._hand_value(self.dealer_hand)
        print(f"Player hand: {self.player_hand} (Total: {player_total}, Usable Ace: {usable})")
        print(f"Dealer hand: {self.dealer_hand} (Total: {dealer_total})")
        print("Card counts:", self.card_counts)
        print("First move:", self.first_move)

# Initialize the new evaluation environment and load the model once at startup.
eval_env = BlackjackEnvCountingFirstMove()
model = PPO.load("models/phase-1-model.zip", env=eval_env)

class PredictBlackjackAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = BlackjackStateSerializer(data=request.data)
        if serializer.is_valid():
            # Extract and convert the blackjack state to a NumPy array.
            # The state should now be in the form:
            # [player_sum, usable_ace, dealer_card, is_first_move, cnt_A, cnt_2, ..., cnt_9, cnt_10group]
            state_list = serializer.validated_data.get("state")
            state = np.array(state_list, dtype=np.int32)
            try:
                action, _ = model.predict(state, deterministic=True)
                return Response({"action": int(action)}, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# views.py
from django.http import JsonResponse

def health(request):
    return JsonResponse({"status": "ok"})

