import torch
import torch.nn as nn
from torch.distributions import Normal

from source.models import ActorCritic
from source.utils import Memory


class PPOAgent:
    def __init__(self, num_actions, learning_rate, gamma, k_epochs, eps_clip, device):
        # --- Hyperparameters ---
        self.lr = learning_rate
        self.gamma = gamma
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.device = device

        # --- The Agent's Brain, moved to the correct device ---
        self.policy = ActorCritic(num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.policy_old = ActorCritic(num_actions).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.memory = Memory()

    def select_action(self, image_obs, text_obs):
        # Convert observation to tensor and move to device
        image_obs = torch.FloatTensor(image_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_params, _ = self.policy_old.forward(image_obs, text_obs)

        action_dist = Normal(action_params, 0.5)
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action).sum()

        # Store tensors on CPU to avoid filling GPU memory
        self.memory.states.append((image_obs.cpu(), text_obs))
        self.memory.actions.append(action.cpu())
        self.memory.logprobs.append(action_logprob.cpu())

        return action.detach().cpu().numpy().flatten()

    def update(self):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize rewards and move to device
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert old data from memory to tensors and move to device
        old_images = torch.squeeze(torch.stack([s[0] for s in self.memory.states], dim=0)).detach().to(self.device)
        old_texts = [s[1][0] for s in self.memory.states]
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).detach().to(self.device)

        # The PPO Update Loop
        for _ in range(self.k_epochs):
            action_params, state_values = self.policy(old_images, old_texts)
            dist = Normal(action_params, 0.5)

            logprobs = dist.log_prob(old_actions).sum(axis=-1)
            dist_entropy = dist.entropy().sum(axis=-1)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach().squeeze()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values.squeeze(), rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear_memory()