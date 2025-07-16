import torch
import os
from datetime import datetime

from environment import OrionEnv
from source.agent import PPOAgent


def train():
    """Main training loop for the Orion agent."""

    ################# Hyperparameters #################
    log_interval = 1
    max_episodes = 50000
    max_timesteps = 300

    update_timestep = 2000
    k_epochs = 40
    eps_clip = 0.2
    gamma = 0.99
    lr = 0.0003

    num_actions = 7

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = "models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #####################################################

    # 1. Initialize environment and agent, passing the device
    env = OrionEnv(render=False)
    agent = PPOAgent(num_actions, lr, gamma, k_epochs, eps_clip, device)

    # --- Logging variables ---
    running_reward = 0
    time_step = 0

    # --- The Main Training Loop ---
    for i_episode in range(1, max_episodes + 1):
        image_obs = env.reset()
        instruction = ["pick up the red cube"]

        for t in range(max_timesteps):
            time_step += 1
            action = agent.select_action(image_obs, instruction)
            image_obs, reward, done, _ = env.step(action)

            agent.memory.rewards.append(reward)
            agent.memory.is_terminals.append(done)

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"orion_model_{timestamp}.pth")
            torch.save(agent.policy.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    env.close()


if __name__ == '__main__':
    train()