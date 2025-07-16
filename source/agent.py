import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models import ActorCritic
from src.utils import Memory

class PPOAgent:
    def __init__(self, num_actions, learning_rate, gamma, k_epochs, eps_clip):
        # --- Hyperparameters ---
        self.lr = learning_rate
        self.gamma = gamma # Discount factor for future rewards
        self.k_epochs = k_epochs # How many times to update the policy per batch
        self.eps_clip = eps_clip # The clipping parameter for the PPO objective

        # --- The Agent's Brain ---
        # We have two identical models. 'policy' is the one we actively train.
        # 'policy_old' is a copy used to calculate the advantage, which remains fixed during an update.
        self.policy = ActorCritic(num_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.policy_old = ActorCritic(num_actions)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss() # For the value function loss
        self.memory = Memory()

    def select_action(self, image_obs, text_obs):
        """
        Selects an action given the current state.
        This is the policy in action.
        """
        # Convert observations to tensors
        image_obs = torch.FloatTensor(image_obs).unsqueeze(0)
        
        with torch.no_grad():
            # Use the 'old' policy to generate the action
            action_params, _ = self.policy_old.forward(image_obs, text_obs)
        
        # --- Stochastic Action Selection ---
        # We create a Normal distribution from the mean output by the network.
        # This allows for exploration. We use a fixed standard deviation.
        action_dist = Normal(action_params, 0.5) # mean, stddev
        action = action_dist.sample()
        action_logprob = action_dist.log_prob(action)
        
        # Store the state, action, and log probability for the upcoming update
        self.memory.states.append((image_obs, text_obs))
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        
        return action.detach().numpy().flatten()

    def update(self):
        """
        Update the policy using the collected trajectory data.
        This is the core of the PPO algorithm.
        """
        # --- Monte Carlo estimate of rewards ---
        # We calculate the "rewards-to-go" for each step in the trajectory.
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalize the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert lists to tensors
        old_images = torch.squeeze(torch.stack([s[0] for s in self.memory.states], dim=0)).detach()
        old_texts = [s[1][0] for s in self.memory.states]
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).detach()

        # --- The PPO Update Loop ---
        for _ in range(self.k_epochs):
            # 1. Evaluate old actions and values using the current policy
            action_params, state_values = self.policy(old_images, old_texts)
            dist = Normal(action_params, 0.5)
            
            # 2. Get the log probabilities of the old actions under the new policy
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            
            # 3. Calculate the ratio (pi_new / pi_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 4. Calculate the surrogate loss (the PPO objective)
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # The final loss is a combination of policy loss, value loss, and entropy bonus
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # 5. Take a gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # --- Copy new weights to old policy ---
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear memory for the next trajectory
        self.memory.clear_memory()
