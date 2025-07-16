# Orion: Language-Conditioned Robotic Manipulation

**Orion** is a deep reinforcement learning framework for training an embodied AI agent to perform robotic manipulation tasks based on natural language commands. The agent processes visual information from a simulated environment and a text-based instruction to execute complex actions, such as picking, placing, and stacking objects.

This project integrates three core AI domains:
* **🤖 Reinforcement Learning:** Utilizes Proximal Policy Optimization (PPO) for stable policy learning in a continuous control environment.
* **👁️ Computer Vision:** Employs a pre-trained ResNet-18 to extract salient features from environmental camera images.
* **🧠 Natural Language Processing:** Uses a pre-trained DistilBERT model from Hugging Face to encode natural language instructions into actionable vector representations.

---
## Demonstration

The agent's goal is to interpret a command, like "place the blue object on the red area," and execute the corresponding actions.

*Note: The following is an illustrative GIF representing the project's objective.*
<p align="center">
  <img src="assets/demo.gif" width="400">
</p>

---
## Environment

The project uses the PyBullet physics simulator. The environment consists of a tabletop, a Kuka IIWA robot arm, and several objects. The agent receives visual input from a simulated camera, which provides RGB, depth, and segmentation data.

<p align="center">
  <img src="assets/environment_pic.png" width="750">
</p>

---

## Architecture

The agent's "brain" is a multi-modal policy network that fuses information from the vision and language modalities before making a decision.

1.  **Environment:** A simulated tabletop scene built using **PyBullet**.
2.  **Vision Module:** A frozen **ResNet-18** processes 128x128 pixel images from the simulator's camera.
3.  **Language Module:** A frozen **DistilBERT** model processes the text command.
4.  **Fusion & Policy:** The vision and language vectors are concatenated and fed into a 2-layer MLP which serves as the shared Actor-Critic network for the PPO algorithm.

---
## Training Results

The model was trained for 5,000 episodes. The average reward per episode shows a clear positive trend, indicating that the agent successfully learned the task policies.

Episode 20      Average Reward: -15.72
Model saved to models/orion_model_20250716_231501.pth
...
Updating policy...
...
Episode 3420    Average Reward: -2.85
Model saved to models/orion_model_20250717_034510.pth
...
Updating policy...
...
Episode 4980    Average Reward: 18.98
Model saved to models/orion_model_20250717_081533.pth
Episode 5000    Average Reward: 21.05
Model saved to models/orion_model_final.pth

---
## Setup & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/Orion-Language-Conditioned-Robotic-Manipulation.git](https://github.com/your-username/Orion-Language-Conditioned-Robotic-Manipulation.git)
cd Orion-Language-Conditioned-Robotic-Manipulation
