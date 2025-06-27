# 🎲 Blackjack RL Agents

This repository contains implementations and training code for Reinforcement Learning agents tackling various Blackjack environments, developed as part of the Master’s thesis by Adarsh Sharma under the supervision of Prof. Prabhat Kumar Mishra in the Department of Artificial Intelligence at IIT Kharagpur.

Till the time of writing this thesis, this work represents one of the first applications of GRPO outside of LLM-focused tasks, and specifically the very first for Blackjack. While GRPO exists in the trl library for language models, there is no implementation for non-LLM settings—not in any popular RL libraries (e.g., Stable Baselines, RLlib), GitHub repositories, or elsewhere on the internet. Adapting GRPO to a Blackjack environment required significant algorithmic changes compared to the LLM setting, so I implemented GRPO and its variants (including DAPO loss) from scratch in Python.

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

## 🛠️ Implementation Details

* **Monte Carlo**: Implemented completely from scratch following first-visit Monte Carlo control.
* **PPO**: Leveraged the Stable Baselines library for Proximal Policy Optimization.
* **GRPO (Group Relative Policy Optimization)**: The core contribution of this thesis—implemented GRPO and its variants from scratch in an extended Blackjack environment and evaluated their performance in a non-LLM setting. For detailed insights and implementation flow, see section 4.2.3.3 (Group Relative Policy Optimization Training) in the full thesis report: [MTP_Thesis_Adarsh_Final.pdf](https://github.com/JaishreeramCoder/blackjack/blob/master/submission-report/MTP_Thesis_Adarsh_Final.pdf).

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

<a id="web-application"></a>
## Web Application 🌐

A web application is built with a React frontend and styled using Tailwind CSS. The FastAPI backend provides real-time predictions from the trained model.  
Access the web app here: [Blackjack Advisor](https://blackjack-beta-kohl.vercel.app/) 🔗

<a id="additional-models--results"></a>

## 🚀 Getting Started

For setup, training, and evaluation instructions, see the detailed READMEs in each model folder:

* `monte-carlo/README.md` 📄
* `ppo/README.md` 📄

**LLM & RLHF Code:**

Code and related resources for the Large Language Model / RLHF component of the thesis are available here:

➡️ [Masters-Thesis-Project1](https://github.com/JaishreeramCoder/Masters-Thesis-Project1)

## 📄 Full Thesis Report

For complete methodology, experiment details, and in-depth discussion of both the RLHF and Blackjack parts of the thesis, see the full report:

➡️ [MTP_Thesis_Adarsh_Final.pdf](https://github.com/JaishreeramCoder/blackjack/blob/master/submission-report/MTP_Thesis_Adarsh_Final.pdf)

---
