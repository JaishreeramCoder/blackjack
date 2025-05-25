# 🎲 Blackjack RL Agents

This repository contains implementations and training code for Reinforcement Learning agents tackling various Blackjack environments, developed as part of the Master’s thesis by Adarsh Sharma under the supervision of Prof. Prabhat Kumar Mishra in the Department of Artificial Intelligence at IIT Kharagpur.

## 🃏 Environments

1. **Simple Blackjack**

   * Actions: Hit / Stand
   * No natural blackjack or doubling-down payouts
2. **Extended Blackjack**

   * Actions: Hit, Stand, Double Down, Surrender
   * Infinite deck (no card counting)
3. **Extended Blackjack with Card Counting**

   * Single deck, with running count tracked in the state
   * Full action set (Hit, Stand, Double Down, Surrender)

## 📈 Models & Performance

| Environment                   | Method                 | Avg. Reward per Bet | Win %       |
| ----------------------------- | ---------------------- | ------------------- | ----------- |
| **Simple Blackjack**          | Monte Carlo            | –4.68 %             | 43.25 %     |
|                               | PPO (Stable Baselines) | –4.57 %             | 43.12 %     |
| **Extended Blackjack**        | Monte Carlo            | –0.45 %             | 41.12 %     |
|                               | PPO                    | –0.99 %             | 38.47 %     |
| **Extended w/ Card Counting** | PPO                    | +2.36 %             | 41.58 %     |
|                               | GRPO + DAPO            | (lower avg. reward) | **44.39 %** |

* Monte Carlo converges quickly in simple settings but struggles with large state spaces.
* PPO excels when card counting is enabled, achieving a +2.36 % average return.
* Group Relative Policy Optimization (GRPO) with DAPO achieves the highest win rate (44.39 %) in the card-counting scenario.

## 🚀 Getting Started

For setup, training, and evaluation instructions, see the detailed READMEs in each model folder:

* `monte-carlo/README.md` 📄
* `ppo/README.md` 📄

## 📄 Full Thesis Report

For complete methodology, experiment details, and in-depth discussion of both the RLHF and Blackjack parts of the thesis, see the full report:

➡️ [MTP\_Thesis\_Adarsh\_Final.pdf](https://github.com/JaishreeramCoder/blackjack/blob/master/submission-report/MTP_Thesis_Adarsh_Final.pdf)

---