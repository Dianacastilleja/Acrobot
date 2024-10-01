import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.buffer)

# Epsilon-greedy action selection with noise
def select_action(state, policy_net, epsilon, env):
    if random.random() < epsilon:
        return env.action_space.sample()  # Exploration: random action
    else:
        # Exploitation: choose the best action, with added noise for exploration
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_tensor)
        action = q_values.max(1)[1].item()

        # Add some random noise to encourage exploration
        noise = np.random.normal(0, 0.1)
        noisy_action = min(max(int(action + noise), 0), env.action_space.n - 1)

        return noisy_action

# Optimize the model
def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < 64:  # Batch size
        return

    states, actions, rewards, next_states, dones = memory.sample(64)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    # Compute Q values
    current_q_values = policy_net(states).gather(1, actions)
    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
    target_q_values = rewards + (0.99 * next_q_values * (1 - dones))

    # Compute loss
    loss = nn.functional.mse_loss(current_q_values, target_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Custom Acrobot Environment
class CustomAcrobotEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env = gym.make("Acrobot-v1", render_mode="rgb_array")  # Using rgb_array for rendering
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, _, done, truncated, info = self.env.step(action)
        custom_reward = self.custom_reward(observation)
        return observation, custom_reward, done, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode)

    # Updated reward function with height and velocity
    def custom_reward(self, observation):
        # Cosine of the first and second link angles (height-related)
        first_link_cos = observation[0]
        second_link_cos = observation[1]

        # Approximate height of the end effector
        height = -(first_link_cos + second_link_cos)

        # Reward for moving the end effector higher
        height_reward = height * 200  # Stronger reward for upward movement

        # Encourage faster upward swinging
        angular_velocity_1 = observation[4]  # Angular velocity of the first link
        angular_velocity_2 = observation[5]  # Angular velocity of the second link
        velocity_reward = (angular_velocity_1 + angular_velocity_2) * 5  # Reduce scaling to avoid instability

        # Small penalty to avoid long episodes without progress
        step_penalty = -1

        # Combine components for final reward
        return height_reward + velocity_reward + step_penalty

# Main training loop
def train_dqn():
    env = CustomAcrobotEnv()
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = ReplayBuffer(10000)
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99995  # Extremely slow decay for more exploration

    writer = SummaryWriter(log_dir='./ppo_acrobot_tensorboard/')  # TensorBoard writer

    num_episodes = 2000  # Increased number of episodes
    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        total_reward = 0
        done = False

        for t in range(1000):
            action = select_action(state, policy_net, epsilon, env)
            next_state, reward, done, truncated, _ = env.step(action)
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model(memory, policy_net, target_net, optimizer)

            if done:
                break

        # Epsilon decay
        epsilon = max(epsilon_min, epsilon_decay * epsilon)

        # Update the target network periodically
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Log progress to TensorBoard
        writer.add_scalar('Reward', total_reward, episode)
        writer.add_scalar('Epsilon', epsilon, episode)

        # Log Q-values for analysis
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(state_tensor)
        writer.add_scalar('Q-Values', q_values.mean().item(), episode)

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # Save the model after training
    torch.save(policy_net.state_dict(), "dqn_acrobot.pth")

    writer.close()  # Close TensorBoard writer
    env.close()

if __name__ == "__main__":
    train_dqn()
