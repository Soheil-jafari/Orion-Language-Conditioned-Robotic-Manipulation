import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertModel, DistilBertTokenizer

# --- Principle: Feature Extractors ---
# We use powerful, pre-trained models (ResNet, DistilBERT) as "feature extractors."
# We freeze their weights because they already know how to understand images and text.
# Our RL agent's job is to learn how to USE these features, not to learn vision or language from scratch.

class VisionModule(nn.Module):
    """
    Processes a batch of images to extract visual features.
    """
    def __init__(self, feature_dim=512):
        super(VisionModule, self).__init__()
        # Load pre-trained ResNet-18
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer to get the feature vector
        self.resnet.fc = nn.Identity()
        
        # Freeze all the parameters in the network
        for param in self.resnet.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Input 'x' is expected to be a batch of images (B, C, H, W)
        # PyTorch models expect channels-first format
        x = x.permute(0, 3, 1, 2)
        return self.resnet(x)

class LanguageModule(nn.Module):
    """
    Processes a batch of text instructions to extract language features.
    """
    def __init__(self, feature_dim=768):
        super(LanguageModule, self).__init__()
        # Load pre-trained DistilBERT model and tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Freeze all the parameters in the network
        for param in self.bert.parameters():
            param.requires_grad = False
            
    def forward(self, instructions):
        # 'instructions' is a list of strings
        inputs = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        # We use the embedding of the [CLS] token as the representation of the whole sentence
        return outputs.last_hidden_state[:, 0, :]

class ActorCritic(nn.Module):
    """
    The main network for the RL agent. It combines vision and language features
    and outputs an action distribution (Actor) and a state value (Critic).
    """
    def __init__(self, num_actions, vision_dim=512, lang_dim=768, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Instantiate the perception modules
        self.vision_module = VisionModule(feature_dim=vision_dim)
        self.language_module = LanguageModule(feature_dim=lang_dim)
        
        # --- Fusion Layer ---
        # This MLP learns to combine the features from both modalities
        self.fusion_layer = nn.Sequential(
            nn.Linear(vision_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # --- Actor Head ---
        # Outputs the parameters for the action distribution (e.g., mean of a Gaussian)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
            nn.Tanh() # Tanh to constrain actions to a [-1, 1] range, can be scaled later
        )
        
        # --- Critic Head ---
        # Outputs a single value representing the "goodness" of the current state
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, image_obs, text_obs):
        # 1. Get features from each modality
        vision_features = self.vision_module(image_obs)
        language_features = self.language_module(text_obs)
        
        # 2. Fuse the features
        fused_features = torch.cat([vision_features, language_features], dim=1)
        fused_output = self.fusion_layer(fused_features)
        
        # 3. Get actor and critic outputs
        action_params = self.actor(fused_output)
        state_value = self.critic(fused_output)
        
        return action_params, state_value

# --- Example of how to use the ActorCritic model ---
if __name__ == '__main__':
    # Define a dummy action space size (e.g., 7 joints for the Kuka arm)
    NUM_ACTIONS = 7 
    
    # Create the model
    model = ActorCritic(num_actions=NUM_ACTIONS)
    
    # Create dummy inputs (batch of 1)
    dummy_image = torch.rand(1, 128, 128, 3) # (B, H, W, C) from PyBullet
    dummy_instruction = ["pick up the red block"]
    
    # Perform a forward pass
    action_parameters, value = model(dummy_image, dummy_instruction)
    
    print("--- Model Test ---")
    print("Action Parameters Shape:", action_parameters.shape)
    print("State Value Shape:", value.shape)
    print("Successfully created and tested the ActorCritic model.")
