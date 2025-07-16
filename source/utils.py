import numpy as np

class Memory:
    """A buffer for storing trajectories experienced by a PPO agent."""
    def __init__(self):
        self.actions = []
        self.states = [] # This will store tuples of (image, text)
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
