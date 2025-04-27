import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import math
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordEpisodeStatistics
import logging
import os

# Set up logging: output will be saved to 'training_log.txt'
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training_log.txt',
                    filemode='w')
logger = logging.getLogger()

# ----------------------------
# Policy Network Definition with Combined Action Selection and Action Masking
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=64, output_dim=4):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

    def _apply_mask(self, state, logits):
        # Masks out double down (index 2) and surrender (index 3) when first-move flag is 0.
        if len(state.shape) == 2:
            first_move_flags = state[:, 3]
            mask = (first_move_flags.unsqueeze(1) == 0).expand_as(logits)
            action_mask = torch.zeros_like(logits, dtype=torch.bool)
            action_mask[:, 2] = True
            action_mask[:, 3] = True
            final_mask = mask & action_mask
            logits = logits.masked_fill(final_mask, -1e9)
        else:
            if state[3] == 0:
                logits[2] = -1e9
                logits[3] = -1e9
        return logits

    def get_log_probs(self, states, actions):
        logits = self.forward(states)
        logits = self._apply_mask(states, logits)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        return log_probs

    def select_action(self, state, deterministic=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        logits = self.forward(state)
        logits = self._apply_mask(state, logits)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
        return action

    def copy(self):
        return copy.deepcopy(self)

def initialize_policy_network():
    return PolicyNetwork(input_dim=14, hidden_dim=64, output_dim=4)

# ----------------------------
# Blackjack Environment with Card Counting & First-Move Flag
# ----------------------------
class BlackjackEnvCountingFirstMove(gym.Env):
    def __init__(self, payout_blackjack=1.5, deck_threshold=15):
        super(BlackjackEnvCountingFirstMove, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([32, 2, 11, 2] + [5]*9 + [17])
        self.payout_blackjack = payout_blackjack
        self.base_bet = 1.0
        self.deck_threshold = deck_threshold
        self._init_deck()
        self.reset()
        
    def _init_deck(self):
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
        dealer_face_up = self.dealer_hand[0]
        dealer_card_val = self._card_value(dealer_face_up)
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
            if dealer_total == 21:
                self.reward = 0.0
            else:
                self.reward = self.payout_blackjack * self.base_bet
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

def initialize_environment():
    return BlackjackEnvCountingFirstMove()

# ----------------------------
# Custom Wrapper to Attach Bet Info
# ----------------------------
class BetInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BetInfoWrapper, self).__init__(env)
    
    def step(self, action):
        self.last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info["bet"] = 2.0 if self.last_action == 2 else 1.0
        return obs, reward, terminated, truncated, info

# ----------------------------
# Custom Callback for Training Metrics
# ----------------------------
class BaseCallback:
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.locals = {}
    def _on_step(self) -> bool:
        return True

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, print_freq=100_000, verbose=0):
        super(TrainingMetricsCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.total_reward = 0.0
        self.total_bet = 0.0
        self.episode_count = 0
        self.episode_history = []
        self.ratio_history = []
    
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        if infos is not None and "episode" in infos:
            self.episode_count += 1
            self.total_reward += float(infos["episode"]["r"])
            self.total_bet += float(infos.get("bet", 1.0))
            if self.episode_count % self.print_freq == 0:
                ratio = (float(self.total_reward) / float(self.total_bet) * 100.0) if self.total_bet > 0 else 0.0
                logger.info(f"Episode: {self.episode_count}, Total Reward: {self.total_reward:.2f}, Total Bet: {self.total_bet:.2f}, Reward/Bet Ratio: {ratio:.2f}%")
                self.episode_history.append(self.episode_count)
                self.ratio_history.append(ratio)
        return True

# ----------------------------
# Cosine Learning Rate Schedule Function
# ----------------------------
def cosine_lr_schedule(progress_remaining):
    lr_initial = 1e-4
    lr_min = 1e-7
    cosine_decay = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
    return lr_min + (lr_initial - lr_min) * cosine_decay

# ----------------------------
# Training Hyperparameters & Initialization
# ----------------------------
batch_size = 64  
num_epochs = 10 
num_episodes = 20_000_000  
G = 8        # Episodes per policy update
miu = 2      # GRPO iterations
epsilon = 0.2
beta = 0.04  # KL penalty coefficient
entropy_coef = 0.01  # (Not used in this version)

cur_policy = initialize_policy_network()
old_policy = cur_policy.copy()
ref_policy = cur_policy.copy()

env = initialize_environment()
env = BetInfoWrapper(env)
env = RecordEpisodeStatistics(env)

training_callback = TrainingMetricsCallback(print_freq=100_000)
optimizer = torch.optim.Adam(cur_policy.parameters(), lr=1e-4)

# Frequency (in episodes) to update the learning rate.
lr_update_freq = 100_000
global_episode = 0
episodes_per_epoch = num_episodes // num_epochs

# ----------------------------
# Updated Policy Update Function (DAPO-style, without entropy bonus)
# ----------------------------
def update_cur_policy(env, cur_policy, old_policy, ref_policy, initial_state, optimizer):
    episodes_states = []
    episodes_actions = []
    episodes_rewards = []
    all_terminal_infos = []
    for _ in range(G):
        state = initial_state.copy()
        states, actions = [], []
        done = False
        while not done:
            action = old_policy.select_action(state, deterministic=False)
            next_state, final_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(state)
            actions.append(action)
            state = next_state
        if len(actions) > 0:
            all_terminal_infos.append(info)
        episodes_states.append(states)
        episodes_actions.append(actions)
        episodes_rewards.append(final_reward)
    final_rewards = np.array(episodes_rewards, dtype=np.float32)
    mean_reward   = np.mean(final_rewards)
    std_reward    = np.std(final_rewards) if np.std(final_rewards) > 1e-8 else 1.0
    batch_states  = []
    batch_actions = []
    batch_advs    = []
    for i in range(G):
        adv_i = (final_rewards[i] - mean_reward) / std_reward
        for s, a in zip(episodes_states[i], episodes_actions[i]):
            batch_states.append(s)
            batch_actions.append(a)
            batch_advs.append(adv_i)
    # Convert list of numpy arrays to a single numpy array for efficiency.
    states_tensor  = torch.tensor(np.array(batch_states), dtype=torch.float32)
    actions_tensor = torch.tensor(batch_actions, dtype=torch.long)
    adv_tensor     = torch.tensor(batch_advs, dtype=torch.float32)
    for _ in range(miu):
        cur_log_probs = cur_policy.get_log_probs(states_tensor, actions_tensor)
        with torch.no_grad():
            old_log_probs = old_policy.get_log_probs(states_tensor, actions_tensor)
            ref_log_probs = ref_policy.get_log_probs(states_tensor, actions_tensor)
        ratio = torch.exp(cur_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        surr1 = ratio * adv_tensor
        surr2 = clipped_ratio * adv_tensor
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        ratio_ref = torch.exp(ref_log_probs - cur_log_probs)
        kl_element = ratio_ref - torch.log(ratio_ref) - 1.0
        kl_div = torch.mean(kl_element)
        loss = policy_loss + beta * kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return all_terminal_infos

# ----------------------------
# Main Training Loop with Epochs and Frequent LR Scheduling & Logging
# ----------------------------
logger.info("Starting training...")
print("Starting training...")
for epoch in range(num_epochs):
    ref_policy = cur_policy.copy()
    for i in range(episodes_per_epoch):
        initial_state, _ = env.reset()
        terminal_infos = update_cur_policy(env, cur_policy, old_policy, ref_policy, initial_state, optimizer)
        for info in terminal_infos:
            if "episode" in info:
                training_callback.locals = {"infos": info}
                training_callback._on_step()
        global_episode += 1
        if global_episode % lr_update_freq == 0:
            progress_remaining = 1.0 - (global_episode / num_episodes)
            new_lr = cosine_lr_schedule(progress_remaining)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"Updated learning rate to {new_lr:.1e} at global episode {global_episode}")
        if (i + 1) % batch_size == 0:
            old_policy = cur_policy.copy()
    logger.info(f"Epoch {epoch+1}/{num_epochs} completed.")
    print(f"Epoch {epoch+1}/{num_epochs} completed.")

logger.info("Training completed.")
print("Training completed.")

# ----------------------------
# Plotting and Saving Training Plot
# ----------------------------
plt.figure(figsize=(10, 5))
plt.plot(training_callback.episode_history, training_callback.ratio_history, marker='o', linestyle='-')
plt.title("Training: Reward/Bet Ratio Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward/Bet Ratio (%)")
plt.grid(True)
training_plot_path = "training_reward_bet_ratio.png"
plt.savefig(training_plot_path)
logger.info(f"Training plot saved to {training_plot_path}")
plt.close()

# ----------------------------
# Evaluation: Use the trained model for evaluation
# ----------------------------
logger.info("Evaluating trained model...")
print("Evaluating trained model...")
cur_policy.eval()
eval_env = initialize_environment()
eval_env = BetInfoWrapper(eval_env)
eval_env = RecordEpisodeStatistics(eval_env)
num_eval_episodes = 100_000
reward_history = []
bet_history = []
ratio_history_eval = []
episode_x = []
wins = 0
draws = 0
losses = 0
logger.info("Starting evaluation...")
print("Starting evaluation...")
cnt = 0
for episode in range(1, num_eval_episodes + 1):
    obs, _ = eval_env.reset()
    episode_reward = 0
    episode_bet = 0
    done = False
    while not done:
        action = cur_policy.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        episode_reward += reward
        episode_bet += info.get("bet", 1)
        done = terminated or truncated
    if episode_reward > 1:
        cnt += 1
    reward_history.append(episode_reward)
    bet_history.append(episode_bet)
    if episode_reward > 0:
        wins += 1
    elif episode_reward == 0:
        draws += 1
    else:
        losses += 1
    if episode % 1000 == 0:
        total_reward = np.sum(reward_history)
        total_bet = np.sum(bet_history)
        ratio = (total_reward / total_bet * 100) if total_bet > 0 else 0.0
        ratio_history_eval.append(ratio)
        episode_x.append(episode)
        logger.info(f"Episode {episode}: Reward/Bet Ratio = {ratio:.2f}%")
        print(f"Episode {episode}: Reward/Bet Ratio = {ratio:.2f}%")

win_percent = wins / num_eval_episodes * 100
draw_percent = draws / num_eval_episodes * 100
loss_percent = losses / num_eval_episodes * 100
avg_reward_eval = (np.sum(reward_history) / np.sum(bet_history) * 100) if np.sum(bet_history) > 0 else 0.0
logger.info("\nFinal Evaluation Results:")
logger.info(f"Win %: {win_percent:.2f}%")
logger.info(f"Draw %: {draw_percent:.2f}%")
logger.info(f"Loss %: {loss_percent:.2f}%")
logger.info(f"Average Reward per Bet (%): {avg_reward_eval:.2f}%")
logger.info(f"cnt {cnt}")
print("\nFinal Evaluation Results:")
print(f"Win %: {win_percent:.2f}%")
print(f"Draw %: {draw_percent:.2f}%")
print(f"Loss %: {loss_percent:.2f}%")
print(f"Average Reward per Bet (%): {avg_reward_eval:.2f}%")
print(f"cnt {cnt}")

plt.figure(figsize=(10, 5))
plt.plot(episode_x, ratio_history_eval, marker='o', linestyle='-')
plt.title("Evaluation: Reward/Bet Ratio Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Reward/Bet Ratio (%)")
plt.grid(True)
eval_plot_path = "evaluation_reward_bet_ratio.png"
plt.savefig(eval_plot_path)
logger.info(f"Evaluation plot saved to {eval_plot_path}")
plt.close()

# ----------------------------
# Save the Trained Model
# ----------------------------
save_path = "policy_model.pth"
torch.save({
    "model_state_dict": cur_policy.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": num_epochs,
}, save_path)
logger.info(f"Model saved to {save_path}")
print(f"Model saved to {save_path}")