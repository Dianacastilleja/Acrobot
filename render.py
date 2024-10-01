import gymnasium as gym
import torch
import torch.nn as nn

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

def select_action(state, model):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32)
        action = model(state).argmax().item()  # Exploit the policy
        return action

def render_agent(model_path, episodes=5):
    # Load the trained model
    env = gym.make("Acrobot-v1", render_mode="human")  # Specify render mode here
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    model = DQN(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Specify map_location for CPU
    model.eval()

    for episode in range(episodes):
        state, info = env.reset(seed=42)  # Initialize the environment
        total_reward = 0
        done = False

        while not done:
            env.render()  # Render the environment
            action = select_action(state, model)  # Select action
            print(f"Action taken: {action}")  # Log the action taken
            next_state, reward, done, truncated, _ = env.step(action)  # Take action
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    env.close()  # Close the environment

if __name__ == "__main__":
    render_agent("dqn_acrobot.pth", episodes=5)
