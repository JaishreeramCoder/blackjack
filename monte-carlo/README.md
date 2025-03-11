# Blackjack Monte Carlo RL Model 🎲

This project implements a Monte Carlo-based Reinforcement Learning (RL) model to learn an optimal strategy for playing a simplified version of Blackjack. The model is trained over millions of episodes using a custom Gymnasium environment and is evaluated using win/loss/draw metrics along with visual strategy charts. 📊

## Overview

The project is divided into three main parts:
1. **Blackjack Environment:** A custom Gymnasium-based environment that simulates a simplified Blackjack game.
2. **Model Training & Evaluation:** A Monte Carlo control method with an epsilon–greedy policy is used to train the agent. Evaluation metrics are computed and plotted.
3. **Strategy Visualization:** The learned strategy is visualized using heatmaps for both models:
    - The extended-action model (with HIT, STK, DBL, and SUR actions) 🃏
    - The simpler blackjack-v1 model (with just HIT and STK actions) 🎴

---

## 1. Available Actions

### Extended–Action Model

- **HIT (0):**  
  Request another card from the dealer. ➕
  
- **STK (1):**  
  Stand (do not take any more cards). ✋
  
- **DBL (2):**  
  Double Down – double your wager, take exactly one additional card, and then stand. 🔥
  
- **SUR (3):**  
  Surrender – forfeit half of your wager and end the game immediately. 🙅

> **Note:** On the first move, all four actions are available. In subsequent moves, only HIT and STK are allowed.

### Simple blackjack-v1 Model

- **HIT (1):**  
  Request another card from the dealer. ➕
  
- **STK (0):**  
  Stand (do not take any more cards). ✋

---

## 2. Reward Structure 💰

The game rewards (or penalties) are based on typical Blackjack outcomes:

- **Blackjack (Natural):**  
  - If the player's first two cards are an Ace and a 10-value card, it is considered a blackjack.
  - **Payout:** 3:2 (e.g., a $10 bet wins $15, for a total return of $25).

- **Regular Win:**  
  - The player beats the dealer without a blackjack.
  - **Payout:** 1:1 (e.g., a $10 bet wins $10, returning $20 in total).

- **Push (Tie):**  
  - The player's hand ties with the dealer's hand.
  - **Payout:** No profit or loss (wager returned).

- **Loss:**  
  - The dealer beats the player or the player busts.
  - **Payout:** The player loses the full wager.

- **Insurance Bet:** *(Not fully modeled)*  
  - When the dealer's upcard is an Ace, an insurance bet can be made.
  - **Payout if dealer has blackjack:** 2:1.
  - **Loss if dealer does not have blackjack:** Loss of the insurance wager.

- **Surrender:**  
  - The player can choose to surrender on the first move.
  - **Payout:** The player forfeits half of the wager.

---

## 3. Code Flow 🔄

### A. Building the Blackjack Environment

- **Implementation:**  
  A custom `BlackjackEnv` class is created using Gymnasium. It defines:
  - **Action Space (Extended–Action Model):**  
    Four discrete actions: HIT, STK, DBL, SUR.
  - **Observation Space (Extended–Action Model):**  
    A tuple containing the player's current sum, the dealer’s upcard, a flag indicating if a usable ace is present, and a flag indicating if it’s the first move.
  - **Observation Space (Simple blackjack-v1 Model):**  
    A tuple containing the player's current sum (4–21), the dealer’s upcard (1–10), and a flag for a usable ace.
  - **Game Logic:**  
    - **Card Drawing & Hand Management:** Functions to draw a card, generate a hand, and calculate the hand's total with consideration for a usable ace.
    - **Player Actions:** The `step` function implements the game dynamics for each possible action.
    - **Dealer Policy:** When the player stands (or doubles), the dealer draws cards until reaching at least 17.

### B. Training and Evaluation

- **Training Process:**
  - **Method:** First–visit Monte Carlo control with an epsilon–greedy policy.
  - **Episodes:** The model is trained over 10 million episodes.
  - **Discount Factor:** 1
  - **Updates:**  
    - For each episode, state–action pairs are stored.
    - After the episode ends, the Q-values are updated based on the final reward.
  - **Tracking Metrics:**  
    - The cumulative loss percentage (total money lost divided by total wagered money) is tracked and plotted against the number of episodes.

- **Evaluation:**
  - **Policy:** A greedy policy (epsilon = 0) is used for evaluation.
  - **Metrics:**  
    - Win percentage (games where the net profit is positive)
    - Loss percentage (games with a net loss)
    - Draw percentage (games where profit is zero)
  - **Graphing:** A graph plots the Average Reward per Bet percentage across all evaluation games played.

### C. Strategy Visualization

- **Extracting the Strategy:**

  #### Extended–Action Model:
  - **State Space:**  
    `[player sum (2–21), dealer sum (1–10 with Ace as 1), usable ace flag (0 or 1), first move flag (0 for subsequent moves, 1 for the first move)]`
    - For **usable ace = 0:** The player sum is calculated directly (values from 2 to 20; 21 is terminal).
    - For **usable ace = 1:** The saved player sum is between 2 and 10, but the actual sum is computed as `saved_sum + 10` (thus displayed as 12–20).
  - **Action Selection:**  
    - **First Move:** Allowed actions are HIT, STK, DBL, and SUR.
    - **Subsequent Moves:** Only HIT and STK are allowed.
  - **Visualization:**  
    The strategy is visualized using heatmaps where rows represent the actual player sum and columns represent the dealer’s showing card (with 1 displayed as “A”). Each cell is annotated with the optimal action. An improved color grading (using the "viridis" palette) enhances visual differentiation. 🎨

  #### Simple blackjack-v1 Model:
  - **State Space:**  
    `[player current sum (4–21), dealer showing card (1–10), usable ace flag (0 or 1)]`
    - For **usable ace = 0:** Player sums range from 4 to 20 (21 is terminal).
    - For **usable ace = 1:** The saved player sum is between 2 and 10, and the actual sum is computed as `saved_sum + 10` (displayed as 12–20).
  - **Action Space:**  
    Only two actions are allowed: HIT and STK.
  - **Visualization:**  
    Two strategy charts are generated (one for each usable ace case) using heatmaps with the improved "viridis" color grading. Rows represent the actual player sum and columns represent the dealer’s card (with 1 shown as "A"). 🎨

- **Strategy Chart Images:**

  - **Extended–Action Model:**
    ![monte-carlo-with-extended-action-space-and-rewards](https://github.com/JaishreeramCoder/blackjack/blob/master/monte-carlo/strategy-chart-visualization/monte-carlo-with-extended-action-space-and-rewards.png)

  - **Simple blackjack-v1 Model:**
    ![monte-carlo-blackjack-v1](https://github.com/JaishreeramCoder/blackjack/blob/master/monte-carlo/strategy-chart-visualization/monte-carlo-blackjack-v1.png)