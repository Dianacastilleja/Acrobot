import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch.utils.tensorboard import SummaryWriter

# Neural Network for the DQN
class DQNAgent(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer to store experiences
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def add(self, experience):
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Epsilon-greedy policy for action selection
def select_action(agent, state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # Random action for exploration
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent(state_tensor)
        return q_values.argmax().item()  # Best action

# Function to apply action scaling
def apply_action(action):
    return action  # Action values are already discrete: [-1, 0, 1]

# Hyperparameters
state_size = 6
action_size = 3
batch_size = 256
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.998  # Slowed down decay rate
learning_rate = 0.0005  # Reduced learning rate
buffer_size = 200000
sync_target_steps = 50
max_episode_length = 500

# Initialize environment, agent, target network, replay buffer, and TensorBoard writer
env = gym.make('Acrobot-v1')
agent = DQNAgent(state_size, action_size)
target_agent = DQNAgent(state_size, action_size)
target_agent.load_state_dict(agent.state_dict())  # Copy weights initially
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(buffer_size)
writer = SummaryWriter()

# Training function with smooth L1 loss
def train(q, q_target, memory, optimizer):
    for _ in range(10):
        # Sample a batch from memory
        batch = memory.sample(batch_size)
        s, a, r, s_prime, done = zip(*batch)

        # Convert to numpy arrays for more efficient tensor creation
        s = torch.tensor(np.array(s), dtype=torch.float32)
        s_prime = torch.tensor(np.array(s_prime), dtype=torch.float32)
        a = torch.tensor(a, dtype=torch.long).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
        done_mask = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        # Compute Q values from current state
        q_out = q(s)
        q_a = q_out.gather(1, a)

        # Compute target Q values using target network
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * (1 - done_mask)

        # Compute loss using Smooth L1 Loss (Huber Loss)
        loss = F.smooth_l1_loss(q_a, target)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Training loop
num_episodes = 10000
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False
    step = 0

    while not done and step < max_episode_length:
        # Select action using epsilon-greedy policy
        action = select_action(agent, state, epsilon)
        next_state, reward, env_done, truncated, _ = env.step(action)

        done = done or env_done or truncated
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        step += 1

        # Perform optimization steps
        if len(replay_buffer) >= batch_size:
            train(agent, target_agent, replay_buffer, optimizer)

        # Sync the target network
        if step % sync_target_steps == 0:
            target_agent.load_state_dict(agent.state_dict())

    # Epsilon decay for exploration-exploitation balance
    if epsilon > epsilon_min:
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    episode_rewards.append(total_reward)
    writer.add_scalar('Reward/train', total_reward, episode)
    print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# Close environment and TensorBoard writer
env.close()
writer.close()
