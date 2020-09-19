import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import QNetwork
from np_deque import np_deque


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, weights_path=None, **impovements):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed

        Other Params
        ======
            ddqn (bool): Whether to use DDQN, default False
            prioritized_replay (bool): Whether to use prioritized replay buffer, default False
            dueling_dqn (bool): Whether to use Dueling DQN, default False
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self._init_improvements(impovements)
        
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # freeze the target network to avoid grad calculations, only update its params manually
        for param in self.qnetwork_target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        if self.prioritized_replay:
            self.criterion = nn.L1Loss(reduce=False)
        else:
            self.criterion = nn.MSELoss()
        
        if weights_path:
            self.load(weights_path)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        if self.prioritized_replay:
            self.memory = ReplayBuffer(state_size, BUFFER_SIZE, BATCH_SIZE)
        else:
            self.memory = ReplayBuffer(state_size, BUFFER_SIZE, BATCH_SIZE, alpha=0)
    
    def _init_improvements(self, impovements):
        for impr_name in ['ddqn', 'prioritized_replay']:
            setattr(self, impr_name, impovements.get(impr_name, False))
            assert isinstance(getattr(self, impr_name), bool), impr_name + ' must be a boolean value'

        if self.prioritized_replay:
            self.beta = impovements.get('beta0', 0.4)
            assert 0 <= self.beta <= 1, "beta0 should be in [0, 1]"

    def load(self, weights_path):
        checkpoint = torch.load(weights_path)
        self.qnetwork_local.load_state_dict(checkpoint['model_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.t_step = checkpoint['t_step']
        self.memory.load_state(checkpoint['memory'])
        print('Loaded the checkpoint with the best avgerage score of ', checkpoint['best_avg_score'])
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self._learn(GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def _learn(self, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        picked_inds, states, actions, rewards, next_states, dones, probabilities = self.memory.sample()

        pred_values = torch.gather(self.qnetwork_local(states), dim=1, index=actions)
        if self.ddqn:
            with torch.no_grad():
                best_actions = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True)
            gt_next_values = torch.gather(self.qnetwork_target(next_states), dim=1, index=best_actions) * (1 - dones)
        else:
            gt_next_values = self.qnetwork_target(next_states).max(dim=1, keepdim=True)[0] * (1 - dones)
        gt_values = rewards + gamma * gt_next_values

        if self.prioritized_replay:
            loss = self.criterion(pred_values, gt_values)
            self.memory.update_priorities(picked_inds, loss.detach().numpy().ravel())
            w = (probabilities*len(self.memory)).pow(-self.beta)
            w /= max(w)
            loss = (w*loss).sqrt()
            loss = loss.mean()

            self.beta = min(self.beta * 1.001, 1.)
        else:
            loss = self.criterion(pred_values, gt_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self._soft_update(TAU)                     

    def _soft_update(self, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def save(self, weights_path, best_avg_score):
        torch.save({
            't_step': self.t_step,
            'model_state_dict': self.qnetwork_local.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'memory': self.memory.state(),
            'best_avg_score': best_avg_score}, 
            weights_path)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, state_size, buffer_size, batch_size, eps=0.01, alpha=0.6):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch

            eps (float): a small value to add to priorities
            alpha (float): a temperature for priorities
        """
        self.s_size = state_size
        self.memory = np_deque(buffer_size, state_size*2 + 4)
        self.batch_size = batch_size

        self.alpha = alpha
        assert 0 <= self.alpha <= 1, "alpha should be in [0, 1]"
        self.eps = eps
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        priority = max(self.memory[:, -1]) if self.memory else 1
        e = [*state, action, reward, *next_state, done, priority]
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        priorities = np.power(self.memory[:, -1], self.alpha)
        probs = priorities/np.sum(priorities)
        picked_inds = np.random.choice(len(self.memory), self.batch_size, replace=False, p=probs)

        states = torch.from_numpy(self.memory[picked_inds, :self.s_size]).float().to(device)
        actions = torch.from_numpy(self.memory[picked_inds, self.s_size:self.s_size + 1]).long().to(device)
        rewards = torch.from_numpy(self.memory[picked_inds, self.s_size + 1:self.s_size + 2]).float().to(device)
        next_states = torch.from_numpy(self.memory[picked_inds, self.s_size + 2:self.s_size*2 + 2]).float().to(device)
        dones = torch.from_numpy(self.memory[picked_inds, self.s_size*2 + 2:self.s_size*2 + 3].astype(np.uint8)).float().to(device)
        probs_tensor = torch.from_numpy(probs[picked_inds].reshape(-1, 1)).float().to(device)

        return (picked_inds, states, actions, rewards, next_states, dones, probs_tensor)
    
    def update_priorities(self, picked_inds, new_priorities):
        """
        Updates priorities of specific samples from replay buffer.

        Params
        ======
            picked_inds (list): indicies of samples to update
            new_priorities (list): new priorities for the picked samples
        """
        self.memory[picked_inds, -1] = new_priorities + self.eps

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def state(self):
        """Returns the replay buffer for saving in a checkpoint."""
        return self.memory[:, :]
    
    def load_state(self, memory):
        """Load the replay buffer from a checkpoint."""
        self.memory.from_array(memory)