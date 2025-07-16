import torch
import os
from datetime import datetime

from src.environment import OrionEnv
from src.agent import PPOAgent

def train():
    """Main training loop for the Orion agent."""
    
    ################# Hyperparameters #################
    log_interval = 20           # Log progress every N episodes
    max_episodes = 50000        # Max number of training episodes
    max_timesteps = 300         # Max timesteps in one episode
    
    update_timestep = 2000      # Update policy every N timesteps
    k_epochs = 40               # Update policy for K epochs
    eps_clip = 0.2              # Clip parameter for PPO
    gamma = 0.99                # Discount factor
    lr = 0.0003                 # Learning rate for actor-critic
    
    # Define the number of actions the robot arm can take (e.g., 7 joints)
    num_actions = 7
    
    # Create the save directory for models if it doesn't exist
    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #####################################################
    
    # 1. Initialize environment and agent
    env = OrionEnv(render=False) # Set to True to watch training
    agent = PPOAgent(num_actions, lr, gamma, k_epochs, eps_clip)
    
    # --- Logging variables ---
    running_reward = 0
    time_step = 0
    
    # --- The Main Training Loop ---
    for i_episode in range(1, max_episodes + 1):
        # For each episode, reset the environment and get the initial state
        image_obs = env.reset()
        
        # TODO: Define your task instruction here.
        # This could be randomized for a more robust agent.
        instruction = ["pick up the red cube"]
        
        for t in range(max_timesteps):
            time_step += 1
            
            # 2. Select an action using the agent's policy
            action = agent.select_action(image_obs, instruction)
            
            # 3. Take the action in the environment
            image_obs, reward, done, _ = env.step(action)
            
            # TODO: Implement your custom reward and done logic.
            # This is the most critical part for successful training.
            # Example reward:
            # - Negative reward for distance to target
            # - Positive reward for grasping object
            # - Large positive reward for completing the task
            
            # 4. Store the experience in the agent's memory
            agent.memory.rewards.append(reward)
            agent.memory.is_terminals.append(done)
            
            # 5. Update the policy
            if time_step % update_timestep == 0:
                print("Updating policy...")
                agent.update()
            
            running_reward += reward
            if done:
                break
        
        # --- Logging ---
        if i_episode % log_interval == 0:
            avg_reward = running_reward / log_interval
            print(f"Episode {i_episode}\tAverage Reward: {avg_reward:.2f}")
            running_reward = 0
            
            # --- Save the model ---
            # You can save the model at intervals
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"orion_model_{timestamp}.pth")
            torch.save(agent.policy.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
    env.close()

if __name__ == '__main__':
    train()
