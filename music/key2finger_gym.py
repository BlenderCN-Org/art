import sys
sys.path.append("../gym")
import gym


# Create the Piano game environment
env = gym.make('Piano-v0')
env.reset()
rewards = []

state = None
# Testing the environment
if 1:
    for _ in range(100):
        env.render()
        state, reward, done, info = env.step(env.action_space.sample()) # take a random action
        rewards.append(reward)
        if done:
            rewards = []
            env.reset()

print(rewards[-20:])

#save_json('animation.json', recorded_sessions)

env.close()
