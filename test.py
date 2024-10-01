import gymnasium as gym

# Create the Acrobot environment
env = gym.make('Acrobot-v1')

# Reset the environment
obs = env.reset()

print("Gymnasium is working, and Acrobot-v1 environment is ready!")
