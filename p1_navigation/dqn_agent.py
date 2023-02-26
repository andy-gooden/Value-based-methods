import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)   # replay buffer size
BATCH_SIZE = 64          # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1.0                # for soft update of target parameters
LR = 5e-5                # learning rate 
LEARN_EVERY = 4          # how often to learn and update weights
UPDATE_TARGET_EVERY = 40 # how often to update target network w soft update
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
MODEL_LAYER1_SIZE = 256
MODEL_LAYER2_SIZE = 256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.t_step = 0                  # Initialize time step (for updating every LEARN_EVERY steps)
        self.l_step = 0                  # init learning step (for updating target weights UPDATE_TARGET_EVERY learn steps)

        # Q-Network
        self.dqn_current = QNetwork(state_size, action_size, seed, MODEL_LAYER1_SIZE, MODEL_LAYER2_SIZE).to(device)
        self.optimizer = optim.Adam(self.dqn_current.parameters(), lr=LR)
        self.dqn_target = QNetwork(state_size, action_size, seed, MODEL_LAYER1_SIZE, MODEL_LAYER2_SIZE).to(device)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every LEARN_EVERY time steps.
        self.t_step = (self.t_step + 1) % LEARN_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self._learn(experiences, GAMMA)

        if done:
            self.eps = max(self.eps_end, self.eps_decay*self.eps)
            
    def get_action(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        # epsilon-greedy action selection
        if random.random() > self.eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            self.dqn_current.eval()
            with torch.no_grad():
                action_values = self.dqn_current(state)
            self.dqn_current.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def save_policy(self, filename):
        """Save agent's policy to file
        Params
        ======
            filename (string): name of file to store policy info
        """
        self.dqn_current.save_to_file(filename)
        
        
    def load_policy(self, filename):
        """Load agent's policy from file
        Params
        ======
            filename (string): name of file which stores policy info
        """
        self.dqn_current.load_from_file(filename)

    def _learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.dqn_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from current model
        Q_expected = self.dqn_current(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.l_step = (self.l_step + 1) % UPDATE_TARGET_EVERY
        if self.l_step == 0:
            self.soft_update(self.dqn_current, self.dqn_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)