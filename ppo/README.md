# Blackjack with Card Counting using Deep Reinforcement Learning 🃏🤖

This repository implements a custom Blackjack environment that integrates card counting and leverages Deep Reinforcement Learning (Deep RL) to train an agent capable of turning the house edge in the player's favor. Traditional methods like Monte Carlo fail in this scenario due to the enormous state space (nearly 1e9 unique states when card counting is enabled). Instead, modern Deep RL techniques such as PPO, DQN, or GRPO are employed to effectively learn optimal strategies. 🎯

## Table of Contents 📖
- [Introduction](#introduction)
- [Environment Description](#environment-description)
- [Training](#training)
  - [Phase 1](#phase-1)
  - [Phase 2 (Final Model)](#phase-2-final-model)
- [Evaluation](#evaluation)
- [Web Application](#web-application)

<a id="introduction"></a>
## Introduction 🚀

In previous experiments using Monte Carlo methods (e.g., in the `blackjack-v1` and `blackjack-with-extended-action-space-and-rewards` models), card counting was not permitted, and the best average reward per bet percentage achieved was **-0.45%**. However, allowing card counting dramatically increases the state space to almost 1e9 unique states, making table-based approaches infeasible. This repository uses Deep RL methods like PPO to overcome the state-space explosion and improve the average reward per bet, ultimately yielding a long-run profitable strategy for the player. 💰

<a id="environment-description"></a>
## Environment Description 📝

The custom environment, `BlackjackEnvWithCounting`, is built using the Gymnasium framework. It extends the classic Blackjack game by incorporating card counting. Key features include:

- **State Vector**:  
  `[player_sum, usable_ace, dealer_card, cnt_A, cnt_2, ..., cnt_9, cnt_10group]`  
  - `player_sum`: Sum of the player's hand (0–31)  
  - `usable_ace`: Indicator if a usable ace is present (0 or 1)  
  - `dealer_card`: Dealer's face-up card (1–10)  
  - `cnt_A, cnt_2, ..., cnt_9`: Count of each card from Ace to 9  
  - `cnt_10group`: Count for the 10-group (10, Jack, Queen, King)  

- **Actions (Discrete 4)**:  
  - **HIT (0)**: Request another card. 🃏  
  - **STK (1)**: Stand. ✋  
  - **DBL (2)**: Double Down (double wager, take one card then stand). 💵  
  - **SUR (3)**: Surrender (forfeit half the wager). 🙇‍♂️  
  _Note: After the first move, only HIT and STK are allowed._

- **Reward Structure**:  
  Rewards follow standard Blackjack rules:
  - Blackjack pays 3:2. 🏆
  - Regular wins pay 1:1.
  - Pushes return 0.
  - Losses incur the bet.
  - Surrender returns -0.5 times the base bet.
  - Double Down outcomes are scaled by a factor of 2.

A custom wrapper, `BetInfoWrapper`, augments the environment by attaching bet information to episode statistics, and the `RecordEpisodeStatistics` wrapper logs performance metrics. 📊

<a id="training"></a>
## Training ⚙️

Training the PPO model required significant computational resources (totaling nearly 20 hours), so the process was split into two phases:

<a id="phase-1"></a>
### Phase 1 ⏱️
- **Timesteps**: 10,000,000 (≈8,000,000 episodes)
- **Learning Rate Schedule**: Utilized a cosine learning rate scheduler decaying from a maximum of 1e-4 to a minimum of 1e-6.
- **Performance**: Achieved an average reward per bet percentage of **0.3%** during evaluation.

<a id="phase-2-final-model"></a>
### Phase 2 (Final Model) 🔥
- **Timesteps**: 20,000,000 (≈15,100,000 episodes)
- **Process**: The phase-1 model was loaded and further trained.
- **Learning Rate Schedule**: Utilized a cosine learning rate scheduler decaying from a maximum of 1e-6 to a minimum of 1e-7.
- **Performance**: The final model achieved an average reward per bet percentage of **1.29%** during evaluation.  
  This indicates that if a player follows the model's strategy, they can expect a long-run profit. 💸

The training code integrates a custom cosine learning rate schedule and logs performance metrics (reward/ bet ratio) over episodes.

<a id="evaluation"></a>
## Evaluation 📈

The evaluation process involves:
- Running the trained model over 100,000 episodes.
- Logging individual episode rewards and bet amounts.
- Calculating win, draw, and loss percentages.
- Plotting the cumulative reward/ bet ratio over time.
- Generating a summary table of key evaluation metrics.

Evaluation scripts also render graphical outputs (plots and tables) and save the results for further analysis. 🖼️

<a id="web-application"></a>
## Web Application 🌐

A Streamlit web application is available for real-time predictions from the trained model.  
Access the web app here: [https://..link] 🔗

To run the app locally:
```bash
streamlit run app.py