import torch
import time
import argparse

from environment import OrionEnv
from source.models import ActorCritic

def evaluate(model_path, instruction):
    """
    Loads a trained agent and evaluates it on a given instruction.
    """
    
    # --- Hyperparameters (should match training) ---
    num_actions = 7
    
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Environment and Model
    env = OrionEnv(render=True)
    
    # Load the ActorCritic model architecture
    policy = ActorCritic(num_actions).to(device)
    
    # Load the saved weights from the file
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        policy.eval() # Set the model to evaluation mode
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Please ensure the path is correct.")
        return
        
    # --- Run One Episode ---
    image_obs = env.reset()
    done = False
    
    print("\n--- Starting Evaluation ---")
    print(f"Instruction: '{instruction[0]}'")

    while not done:
        # Convert observation to tensor and move to device
        image_obs_tensor = torch.FloatTensor(image_obs).unsqueeze(0).to(device)
        
        # 2. Select a deterministic action (no random sampling)
        with torch.no_grad():
            action_params, _ = policy(image_obs_tensor, instruction)
        
        # The output of the actor is the mean of the action distribution
        action = action_params.cpu().numpy().flatten()
        
        # 3. Step the environment
        image_obs, reward, done, _ = env.step(action)
        
        # Allow time for rendering
        time.sleep(1./60.) 
        
    print("--- Evaluation Finished ---")
    env.close()

if __name__ == '__main__':
    # --- Command Line Argument Parser ---
    parser = argparse.ArgumentParser(description="Evaluate a trained Orion agent.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.pth) file."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        required=True,
        help="The natural language instruction for the agent."
    )
    args = parser.parse_args()

    evaluate(model_path=args.model_path, instruction=[args.instruction])
