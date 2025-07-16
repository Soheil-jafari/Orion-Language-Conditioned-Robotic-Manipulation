# Orion: Language-Conditioned Robotic Manipulation

**Orion** is a deep reinforcement learning framework for training an embodied AI agent to perform robotic manipulation tasks based on natural language commands. The agent processes visual information from a simulated environment and a text-based instruction to execute complex actions, such as picking, placing, and stacking objects.

This project integrates three core AI domains:
* **🤖 Reinforcement Learning:** Utilizes Proximal Policy Optimization (PPO) for stable policy learning in a continuous control environment.
* **👁️ Computer Vision:** Employs a pre-trained ResNet-18 to extract salient features from environmental camera images.
* **🧠 Natural Language Processing:** Uses a pre-trained DistilBERT model from Hugging Face to encode natural language instructions into actionable vector representations.

![Orion Demo GIF](https://your-link-to-a-demo-gif.com/demo.gif)
*(You will create this GIF once the project is working)*

---

## Architecture

The agent's "brain" is a multi-modal policy network that fuses information from the vision and language modalities before making a decision.

1.  **Environment:** A simulated tabletop scene built using **PyBullet**, containing a Kuka IIWA robot arm and several primitive objects (cubes, spheres) of varying colors.
2.  **Vision Module:** A frozen **ResNet-18** processes 128x128 pixel images from the simulator's camera, producing a `512-dimensional` feature vector.
3.  **Language Module:** A frozen **DistilBERT** model processes the text command, producing a `768-dimensional` embedding.
4.  **Fusion & Policy:** The vision and language vectors are concatenated and fed into a 2-layer MLP (Multi-Layer Perceptron) which serves as the shared Actor-Critic network for the PPO algorithm. The network outputs an action distribution and a state-value estimate.

<p align="center">
  <img src="https://your-link-to-an-architecture-diagram.com/arch.png" width="750">
</p>
*(You can create this diagram easily using tools like diagrams.net)*

---

## How it Works

The training loop follows a standard on-policy RL procedure:
1.  **Instruction:** A task is generated, e.g., "pick up the red block."
2.  **Observe:** The agent receives the instruction and the current camera image of the scene.
3.  **Process:** The vision and language modules generate their respective embeddings.
4.  **Act:** The fused embeddings are passed to the PPO policy, which samples an action (e.g., move arm joint `x` by `y` degrees).
5.  **Learn:** The agent executes the action, receives a reward (based on distance to the target, grasp success, etc.), and stores this experience. The PPO algorithm updates the policy network after collecting a batch of experiences.

---

## Setup & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/Orion.git](https://github.com/your-username/Orion.git)
cd Orion
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Train the agent:**
```bash
python train.py --task "pick_and_place" --epochs 5000
```

**4. Run a trained model:**
```bash
python evaluate.py --model_path "models/orion_final.pth" --instruction "place the blue sphere on the red block"
```
