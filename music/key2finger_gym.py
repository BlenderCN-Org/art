import json
import re
import numpy as np
import random as rd
import tensorflow as tf

import sys
sys.path.append("../gym")
import gym

import matplotlib.pyplot as plt

from collections import deque


def one_hot_decoder(state):
    key_count = 88
    finger_positions = []
    for fn in range(0, 10):
        start = fn*key_count
        end = start+key_count
        pos = 0
        for idxx in range(start, end):
            if state[idxx] == 1.0:
                pos = idxx-start
                finger_positions.append(pos)
    return finger_positions


def get_action(state):
    finger_positions = one_hot_decoder(state)
    lower_key = None
    higher_key = None
    shift = len(state)-(88*2)
    for fn in range(0, 88):
        if state[fn+shift]==1:
            if fn < 44:
                if lower_key is None:
                    lower_key = fn
                if fn < lower_key:
                    lower_key = fn
            elif fn >=44:
                if higher_key is None:
                    higher_key = fn
                if fn > higher_key:
                    higher_key = fn
    actions = []

    if lower_key is not None:
        if finger_positions[2] > lower_key:
            actions.append(12)
        elif finger_positions[2] < lower_key:
            actions.append(2)
    if higher_key is not None:
        if finger_positions[7] > higher_key:
            actions.append(17)
        elif finger_positions[7] < higher_key:
            actions.append(7)

    #
    for finger in range(0, 10):
        # If the finger is already on a key don't move it
        # If the finger is the Middle don't move it
        if state[finger_positions[finger]] == 1 or finger == 2 or finger == 7:
            continue
        for step in range(-3, 3):
            if state[finger_positions[finger]+step] == 1:
                best_action = None
                if step == -3:
                    best_action = 50
                elif step == 3:
                    best_action = 40
                elif step == -2:
                    best_action = 30
                elif step == 2:
                    best_action = 20
                elif step == -1:
                    best_action = 10
                elif step == 1:
                    best_action = 0
                if best_action is not None:
                    actions.append(best_action+finger)

    if len(actions)==0:
        return 0
    else:
        return rd.choice(actions)


# Create the Piano game environment
env = gym.make('Piano-v0')
env.reset()
rewards = []

state = None
# Testing the environment
if 1:
    for _ in range(100):
        env.render()
        if state is None:
            state, reward, done, info = env.step(env.action_space.sample()) # take a random action
        else:
            state, reward, done, info = env.step(get_action(state)) # take a random action
        rewards.append(reward)
        if done:
            rewards = []
            env.reset()

print(rewards[-20:])

class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10,
                 name='QNetwork'):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            #self.inputs_ = tf.placeholder(tf.float32, [state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

            if 0:
                # ReLU hidden layers
                self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
                self.fc2 = tf.nn.dropout(tf.contrib.layers.fully_connected(self.fc1, hidden_size), self.keep_prob)
                self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)
            else:
                self.inputs2d = tf.reshape(self.inputs_, [self.batch_size, 12, 88, 1]) # batch_size * 88*12
                # 1D "image" 88 pixels width, and 12 channels
                self.inputs2d = tf.transpose(self.inputs2d, perm=[0, 2, 3, 1])
                #print (self.inputs2d.shape)
                #lala()

                filters_count = 64
                kernel_size = [4, 1] # Width and Height
                strides = [1, 2] # Height and Width
                self.cv1 = tf.layers.conv2d(self.inputs2d, filters_count, kernel_size, strides, activation=tf.nn.relu6)

                filters_count = 128
                kernel_size = [4, 1] # Width and Height
                strides = [1, 2] # Height and Width
                self.cv2 = tf.layers.conv2d(self.cv1, filters_count, kernel_size, strides, activation=tf.nn.relu6)

                self.fc1 = tf.contrib.layers.flatten(self.cv2)
                self.fc2 = tf.contrib.layers.fully_connected(self.fc1, hidden_size)
                self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length of possible actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded action.
            # Example: [1.2, 3.4, 9.3] x [1, 0, 0] = [1.2, 0, 0] = 1.2
            self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]


def save_json(json_path, data):
    with open(json_path, 'w', encoding='utf-8') as anim_file:
        json.dump(data, anim_file, sort_keys=True, indent=4, separators=(',', ': '))


train_episodes = 10000         # max number of episodes to learn from
training_steps = 10
max_steps = 10000              # max steps in an episode
gamma = 0.99                   # future reward discount

# Exploration parameters
explore_start = 0.99           # exploration probability at start
explore_stop = 0.001           # minimum exploration probability
decay_rate = 0.000001          # exponential decay rate for exploration prob

# Network parameters
hidden_size = 1024             # number of units in each Q-network hidden layer
learning_rate = 0.0000001      # Q-network learning rate

# Memory parameters
memory_size = 1000000          # memory capacity
batch_size = 32                # experience mini-batch size
pretrain_length = 50000        # number experiences to pretrain the memory

do_training = True

tf.reset_default_graph()
key_count = 88
fingers_count = 10
state_size = (key_count * fingers_count) + (key_count * 2) # One hot fingers positions + current and future keyboard status
action_size = (6 * fingers_count) + 1 # 6 actions + "still" action
print ("State Size: {}, Action Size: {}".format(state_size, action_size))
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, state_size=state_size, action_size=action_size)

# Initialize the simulation
env.reset()
# Take one random step to generate an initial state
state, reward, done, _ = env.step(env.action_space.sample())

memory = Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
print ("Initial experiences")
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    #env.render()

    # Make a random action
    #action = env.action_space.sample()
    if 0.5 > np.random.rand():
        action = env.action_space.sample() # take a random action
    else:
        action = get_action(state) # take a predefined action
    next_state, reward, done, _ = env.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to generate an initial state
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

saver = tf.train.Saver()

loss = 0.0
if do_training:
    print ("Training")
    # Now train with experiences
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                #print (len(memory.buffer))
                # Uncomment this next line to watch the training
                if ep % 10 == 0:
                    env.render()

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                if explore_p > np.random.rand():
                    # Make a random action
                    if state is None or 0.95 > np.random.rand():
                        action = env.action_space.sample() # take a random action
                    else:
                        action = get_action(state) # take a predefined action
                else:
                    # Get action from Q-network
                    feed = {
                        mainQN.inputs_: state.reshape((1, *state.shape)),
                        mainQN.batch_size: 1,
                        mainQN.keep_prob: 1.0
                    }
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)

                    if ep % 100 == 0:
                        saver.save(sess, "checkpoints/cartpole.ckpt")

                    if ep % 1 == 0:
                        print('Episode: {}'.format(ep),
                              'Step: {:04d}'.format(t),
                              #'Action: {:.4f}'.format(action),
                              #'Finger position: {:.4f}'.format(env.current_finger_position),
                              'Mean reward: {:.4f}'.format(total_reward/step),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_p))
                    t = max_steps
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()
                    # Take one random step, get new state and reward
                    state, reward, done, _ = env.step(env.action_space.sample())

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                #for _ in range(0, training_steps):
                #if step % training_steps == 0:

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network #
                # Executes batch_size actions, and caches the output
                # of the Neural Network. output = Q
                feed_dict = {
                    mainQN.inputs_: next_states,
                    mainQN.batch_size: batch_size,
                    mainQN.keep_prob: 1.0
                }
                target_Qs = sess.run(mainQN.output, feed_dict=feed_dict)

                data = sess.run(mainQN.inputs2d, feed_dict=feed_dict)
                #save_json('log.json', data.tolist())
                #lala()

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                target_Qs[episode_ends] = [0 for _ in range(0,action_size)]

                # Updates the already generated Q according to the reward
                # like if the generated Q where real (?)
                targets = rewards + gamma * np.max(target_Qs, axis=1)

                # Force the network to output the new Q
                # given the same state and action as before
                feed_dict = {
                    mainQN.inputs_: states,
                    mainQN.targetQs_: targets,
                    mainQN.actions_: actions,
                    mainQN.batch_size: batch_size,
                    mainQN.keep_prob: 7.0
                }
                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict=feed_dict)

        saver.save(sess, "checkpoints/cartpole.ckpt")





    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N


    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


## Testing


test_episodes = 10
test_max_steps = 10000
env.reset()
#env.training = False
recorded_sessions = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for ep in range(1, test_episodes):
        t = 0
        recorded_frames = []
        while t < test_max_steps:
            env.render()

            # Get action from Q-network
            feed = {
                mainQN.inputs_: state.reshape((1, *state.shape)),
                #mainQN.inputs_: state.reshape((1, 12, 88, 1)),
                mainQN.batch_size: 1,
                mainQN.keep_prob: 1.
            }
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
            #finger_position = env.current_finger_position
            #finger_position = 0

            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                # Convert to Python list for Json compatibility and append
                #recorded_frames.append([finger_position, np.asscalar(action)] + state.tolist()[env.key_count:])
                t += 1
        recorded_sessions.append(recorded_frames)

save_json('animation.json', recorded_sessions)

env.close()
