import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from collections import namedtuple, deque
import torch.optim as optim
import numpy as np
import os
import warnings

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-1              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay
LEARN_EVERY = 10        # Only update networks every nth step
NUM_UPDATE = 10         # When updating do it n times
SOFT_UPDATE_EVERY = 1   # Soft update target every nth time
SEED = 7                # Random seed
LOSS_FUNCTION = torch.nn.MSELoss() #torch.nn.SmoothL1Loss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[512,256]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (list of ints): Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = []
        self.fc.append(nn.Linear(state_size, fc_units[0]))
        self.fc.append(nn.ReLU())
        for i in range(1,len(fc_units)):
            self.fc.append(nn.Linear(fc_units[i-1], fc_units[i]))
            self.fc.append(nn.ReLU())
        self.fc.append(nn.Linear(fc_units[-1], action_size))
        self.fc.append(nn.Tanh())
        self.fc = nn.Sequential(*self.fc)
        self.reset_parameters()

    def reset_parameters(self):
        tmp_ = []
        for m_ in self.fc:
            if isinstance(m_,torch.nn.Linear):
                tmp_.append(m_)
        for m_ in tmp_[:-1]:
            m_.weight.data.uniform_(*hidden_init(m_))
        tmp_[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc(state)
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=[512,256],dropout=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Sequential(nn.Linear(state_size, fc_units[0]),
                                 nn.ReLU())
        self.fc2 = []
        for i in range(1,len(fc_units)):
            if i==1:
                self.fc2.append(nn.Linear(fc_units[i-1]+action_size, fc_units[i]))
            else:
                self.fc2.append(nn.Linear(fc_units[i-1], fc_units[i]))
            self.fc2.append(nn.ReLU())
            if dropout is not None:
                self.fc2.append(nn.Dropout(dropout))
        self.fc2.append(nn.Linear(fc_units[-1], 1))
        self.fc2 = nn.Sequential(*self.fc2)
        self.reset_parameters()

    def reset_parameters(self):
        tmp_ = []
        for m_ in self.fc1:
            if isinstance(m_,torch.nn.Linear):
                tmp_.append(m_)
        for m_ in tmp_[:-1]:
            m_.weight.data.uniform_(*hidden_init(m_))
        for m_ in self.fc2:
            if isinstance(m_,torch.nn.Linear):
                tmp_.append(m_)
        for m_ in tmp_[:-1]:
            m_.weight.data.uniform_(*hidden_init(m_))
        tmp_[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fc1(state)
        x = torch.cat((xs, action), dim=1)
        return self.fc2(x)


class DDPG_AGENT():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed,loss_function=LOSS_FUNCTION,
                 batch_size=BATCH_SIZE,learn_every=LEARN_EVERY,gamma=GAMMA,num_update=NUM_UPDATE,
                 soft_update_every=SOFT_UPDATE_EVERY,buffer_size=BUFFER_SIZE,
                 tau=TAU,lr_actor=LR_ACTOR,lr_critic=LR_CRITIC,weight_decay=WEIGHT_DECAY,
                 memory=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.soft_update_every = soft_update_every
        self.gamma = gamma
        self.num_update = num_update
        self.buffer_size = buffer_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.weight_decay = weight_decay

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.loss_function = loss_function

        # Replay memory
        if memory is None:
            self.memory = PrioritizedReplayBuffer(self.buffer_size, self.batch_size, random_seed)
        else:
            self.memory = memory(self.buffer_size, self.batch_size, random_seed)

        self.learn_step = 0
        self.update_step = 0
        self._copy_weights(self.actor_local, self.actor_target)
        self._copy_weights(self.critic_local, self.critic_target)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.append_sample(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        self.update_step += 1
        if (len(self.memory) > self.batch_size) and (self.update_step % self.learn_every == 0):
            for _ in range(self.num_update):
                self.learn()

    def append_sample(self, state, action, reward, next_state, done):

        if isinstance(self.memory,PrioritizedReplayBuffer):
            self.actor_target.eval()
            self.critic_target.eval()
            self.critic_local.eval()

            state_ = torch.from_numpy(state[np.newaxis,:]).float().to(device)
            next_state_ = torch.from_numpy(next_state[np.newaxis,:]).float().to(device)
            action_ = torch.from_numpy(action[np.newaxis,:]).float().to(device)

            with torch.no_grad():
                # Expected next actions from Actor target network based on state
                actions_next = self.actor_target(state_)
                # Predicted Q value from Critic target network
                Q_targets_next = self.critic_target(next_state_, actions_next)
                Q_targets = reward + (self.gamma * Q_targets_next * (1 - done))

                # Actual Q value based on reward rec'd at next step + future expected reward from Critic target network
                Q_expected = self.critic_local(state_, action_)

            error = (Q_expected - Q_targets) ** 2
            self.memory.add(error,(state, action, reward, next_state, done))

            self.critic_local.train()
            self.actor_target.train()
            self.critic_target.train()

        elif isinstance(self.memory, UniformReplayBuffer):
            self.memory.add(state, action, reward, next_state, done)

        else:
            raise ValueError('Only UniformReplayBuffer and PrioritizedReplayBuffer have been implemented.')


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        if isinstance(self.memory, PrioritizedReplayBuffer):
            experiences, idxs, is_weight = self.memory.sample()
        else:
            experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)

        if isinstance(self.memory, PrioritizedReplayBuffer):
            errors =torch.pow((Q_expected - Q_targets),2).data.cpu().numpy().reshape(is_weight.shape)*is_weight
            for i in range(self.batch_size):
                self.memory.update(idxs[i], errors[i])

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.learn_step += 1
        if self.learn_step % self.soft_update_every == 0:
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _copy_weights(self, local_model, target_model):
        """Copy source network weights to target"""
        self.soft_update(local_model,target_model,1)


class RandomNoise:
    def __init__(self, size, seed, scale=1.0):
        self.size = size
        self.scale = scale
        self.seed = random.seed(seed)

    def reset(self):
        pass

    def sample(self):
        return self.scale * np.random.randn(self.size)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2, scale = 1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.scale = scale
        self.state = None
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state*self.scale


class UniformReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
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

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device) #.requires_grad_(False)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device) #.requires_grad_(False)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device) #.requires_grad_(False)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device) #.requires_grad_(False)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device) #.requires_grad_(False)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



class PrioritizedReplayBuffer:

    def __init__(self, buffer_size, batch_size, seed):
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.seed = random.seed(seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, self.experience(*sample))

    def sample(self):
        batch = []
        idxs = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        num_errors = 0
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if isinstance(data,self.experience):
                priorities.append(p)
                batch.append(data)
                idxs.append(idx)
            else:
                num_errors += 1

        if num_errors>0:
            warnings.warn('Prioritized Replay returned {0} problematic samples'.format(num_errors))
            for j in random.sample(range(len(batch)), self.batch_size - len(batch)):
                priorities.append(priorities[j])
                batch.append(batch[j])
                idxs.append(idxs[j])


        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        states = torch.from_numpy(np.vstack([e.state for e in batch])).float().to(device)  # .requires_grad_(False)
        actions = torch.from_numpy(np.vstack([e.action for e in batch])).float().to(device)  # .requires_grad_(False)
        rewards = torch.from_numpy(np.vstack([e.reward for e in batch])).float().to(device)  # .requires_grad_(False)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in batch])).float().to(device)  # .requires_grad_(False)
        dones = torch.from_numpy(np.vstack([e.done for e in batch]).astype(np.uint8)).float().to(device)

        return (states,actions,rewards,next_states,dones), idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.n_entries


def set_all_fontsizes(ax, fontsize, legend=True):

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
    if legend:
        ax.legend(fontsize=fontsize)

    for tick in ax.xaxis.get_minorticklabels():
        tick.set_fontsize(fontsize)