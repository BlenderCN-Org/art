#import gym
import json
import re
import numpy as np
import random as rd
import tensorflow as tf


#%matplotlib inline
import matplotlib.pyplot as plt

from collections import deque


## Environment


def open_midi(midigram_filepath):
    # On (MIDI pulse units)
    # Off (MIDI pulse units)
    # Track number
    # Channel number
    # Midi pitch -> 60 -> c4 / 61 -> c#4
    # Midi velocity
    # 1926 = 4.01 sec
    # 480.299 x sec
    # 16 x frame (30fps)
    midi_pulse_per_second  = 480.299
    fps = 30
    fps = 11.7
    margin = 1
    keyboard_size = 88
    max_frames = 10000

    midi_pulse_per_frame = int(midi_pulse_per_second / fps)

    with open (midigram_filepath, "r") as midigram_file:
        data=midigram_file.readlines()

    last_frame = 0
    lower_pitch = None
    midi_frames = {}
    for data_line in data[:-1]:
        line_regex = re.search(r'^(?P<ontime>\d+)\s(?P<offtime>\d+)\s(?P<track>\d+)\s(?P<channel>\d+)\s(?P<pitch>\d+)\s(?P<velocity>\d+)', data_line)
        ontime = int(int(line_regex.group("ontime"))/midi_pulse_per_frame)
        offtime = int(int(line_regex.group("offtime"))/midi_pulse_per_frame)
        track = int(line_regex.group("track"))
        channel = int(line_regex.group("channel"))
        pitch = int(line_regex.group("pitch"))
        velocity = int(line_regex.group("velocity"))

        if not ontime in midi_frames:
            midi_frames[ontime] = []
        midi_frames[ontime].append({
            "ontime": ontime,
            "offtime": offtime,
            "track": track,
            "channel": channel,
            "pitch": pitch,
            "velocity": velocity,
        })
        if offtime > last_frame:
            last_frame = offtime
        if lower_pitch == None:
            lower_pitch = pitch
        if pitch < lower_pitch:
            lower_pitch = pitch

    music = []
    for frame_number in range(0, last_frame+1):
        if frame_number > max_frames:
            break
        notes = [0 for _ in range(0, keyboard_size)]
        music.append(notes)

    for mf in midi_frames:
        #print (midi_frames[mf])
        #print (midi_frames[mf]['ontime'])
        for mf_note in midi_frames[mf]:
            # Normalize pitch on the keyboard
            mf_normalized_pitch = mf_note['pitch']-lower_pitch
            if mf_normalized_pitch >= keyboard_size:
                print ("Skipping note, out of keyboard range")
                continue
            # Fill ontime to offtime values
            for f in range(mf_note['ontime'], mf_note['offtime']):
                #music[f][mf_normalized_pitch] = ((f-mf_note['ontime']) / (mf_note['offtime']-mf_note['ontime']) * 0.5) + 0.5
                music[f][mf_normalized_pitch] = 1.0
            # Search previous offtime
            previous_offtime = 0
            for mf_back in midi_frames:
                #if mf_back == mf:
                #    break
                for mf_back_note in midi_frames[mf_back]:
                    if mf_back_note['pitch'] == mf_note['pitch']:
                        if mf_back_note['offtime'] <= mf_note['ontime'] and mf_back_note['offtime'] > previous_offtime:
                            previous_offtime = mf_back_note['offtime']
            # Fill previous_offtime to ontime values
            #if mf_note['pitch'] == 64:
            #    print ("Previous offtime {} for ontime {} ({}) note {}".format(previous_offtime, mf_note['ontime'], mf_note['offtime'], mf_note['pitch']))
            #if previous_offtime != mf_note['ontime']:
            #    for f in range(previous_offtime, mf_note['ontime']):
            #        music[f][mf_normalized_pitch] = (f-previous_offtime) / (mf_note['ontime']-previous_offtime+1) * 0.5

            #ontime == offtime
            #offtime == ontime

    notes = [0 for _ in range(0, keyboard_size)]
    music = [notes for _ in range(0, margin)] + music + [notes for _ in range(0, margin)]
    return music


class ActionSpace():
    def __init__(self):
        pass

    def sample(self):
        return rd.randint(0, 6)


class PianoEnv:
    def __init__(self, name):
        self.env_name = name
        self.action_space = ActionSpace()
        #frames_per_second = 60
        #self.tau = 1.0 / frames_per_second
        #music = [[100, 200, 0], [300, 400, 1], [500, 600, 2],]


        self.frames = open_midi('Korobushka_Tetris_Theme.midigram')
        #self.frames = open_midi('test.midigram')

        self.key_count = 88
        self.frame_count = len(self.frames)
        print ("Frame count")
        print (self.frame_count)

        self.reset()
        #print (self.frames)
        #lala()
        #self.length = 600


    def reset(self):
        self.current_frame = rd.randint(0, self.frame_count-1)
        self.current_frame = 0
        current_key = int(self.key_count/2)
        max_note_val = None
        for note_pitch, note_val in enumerate(self.frames[self.current_frame]):
            if max_note_val == None:
                current_key = note_pitch
                max_note_val = note_val
            if note_val > max_note_val:
                current_key = note_pitch
                max_note_val = note_val
        self.current_finger_position = self.key2continous(current_key)
        self.failed_frames = 0
        #print ("## RESET ##")
        #print ('Current frame: {}'.format(self.current_frame),
        #       'Current finger position {}'.format(self.current_finger_position))

    def close(self):
        pass

    def render(self):
        pass

    def key2continous(self, current_key):
        return current_key
        #key_size = 1.0/(self.key_count-1)
        #return (key_size*current_key) + (key_size*0.5)

    def continous2key(self, current_finger_position):
        #key = int(current_finger_position * self.key_count)
        key = current_finger_position
        if key < 0 or key > self.key_count-1:
            key = None
        return key

    def step(self, action):
        # self.tau = 0.02 # seconds between state updates
        # thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))

        # x  = x + self.tau * x_dot
        # x_dot = x_dot + self.tau * xacc
        # theta = theta + self.tau * theta_dot
        # theta_dot = theta_dot + self.tau * thetaacc
        # (x,x_dot,theta,theta_dot)


        #action_step = (1.0/(self.key_count))*0.3
        action_step = 1
        step_sizes = [0, 1, -1, 2, -2, 3, -3]
        #print (action)

        self.current_finger_position += action_step * step_sizes[action]

        #print (self.current_finger_position)

        on_key = self.continous2key(self.current_finger_position)
        one_hot = [0 for _ in range(0, self.key_count)]
        if on_key != None:
            one_hot[on_key] = 1

        key_value = 0.0
        if on_key != None:
            try:
                key_value = self.frames[self.current_frame][on_key]
            except:
                key_value = 0.01
            if key_value == 1.0:
                reward = 0.5
            else:
                reward = 0.01
        else:
            reward = 0.01

        done = False
        must_reset = False

        # If no frames left, done
        if self.current_frame >= len(self.frames)-1-1:
            must_reset = True
            state = np.array(one_hot + self.frames[self.current_frame-1] + self.frames[self.current_frame-1])
        else:
            state = np.array(one_hot + self.frames[self.current_frame] + self.frames[self.current_frame + 1])


        if not must_reset:
            # If the finger falls outside the keyboard, done
            if self.current_finger_position < 0 or self.current_finger_position > self.key_count:
                reward = 0.0001
                must_reset = True

        if not must_reset:
            # If a key should be pressed but is not, done
            #for key, key_val in enumerate(self.frames[self.current_frame]):
            try:
                key = self.frames[self.current_frame].index(1)
            except ValueError:
                key = None
            #if key_val == 1.0 and key != on_key:
            if key != None and key != on_key:
                reward = 0.0001
                self.failed_frames += 1
                if self.failed_frames > 30:
                    must_reset = True

        self.current_frame += 1

        if must_reset:
            done = True
            #self.reset()

        info = {}
        #print ("Eval")
        #print (action)
        #print (state)
        #print (reward)
        #print (done)
        return state, reward, done, info




## Q-Learning

# Create the Cart-Pole game environment
#env = gym.make('CartPole-v0')
env = PianoEnv('CartPole-v0')
#print (env.frames)
#lala()

env.reset()
rewards = []
for _ in range(100):
    env.render()
    state, reward, done, info = env.step(env.action_space.sample()) # take a random action
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

            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.nn.dropout(tf.contrib.layers.fully_connected(self.fc1, hidden_size), 1.0)
            self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size,
                                                            activation_fn=None)

            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
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

#if 0:
train_episodes = 10000         # max number of episodes to learn from
max_steps = env.frame_count    # max steps in an episode
gamma = 0.9                    # future reward discount

# Exploration parameters
explore_start = 0.99            # exploration probability at start
explore_stop = 0.001            # minimum exploration probability
decay_rate = 0.00001            # exponential decay rate for exploration prob

# Network parameters
hidden_size = 128              # number of units in each Q-network hidden layer
learning_rate = 0.00001        # Q-network learning rate

# Memory parameters
memory_size = 100000           # memory capacity
batch_size = 100               # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory


tf.reset_default_graph()
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, state_size=env.key_count*3, action_size=7)

# Initialize the simulation
env.reset()
# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())

memory = Memory(max_size=memory_size)

# Make a bunch of random actions and store the experiences
for ii in range(pretrain_length):
    # Uncomment the line below to watch the simulation
    # env.render()

    # Make a random action
    action = env.action_space.sample()
    next_state, reward, done, _ = env.step(action)

    if done:
        # The simulation fails so no next state
        next_state = np.zeros(state.shape)
        # Add experience to memory
        memory.add((state, action, reward, next_state))

        # Start new episode
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())
    else:
        # Add experience to memory
        memory.add((state, action, reward, next_state))
        state = next_state

saver = tf.train.Saver()

loss = 0.0
do_training = False
if do_training:
    # Now train with experiences
    rewards_list = []
    with tf.Session() as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())

        step = 0
        for ep in range(1, train_episodes):
            total_reward = 0
            t = 0
            while t < max_steps:
                step += 1
                # Uncomment this next line to watch the training
                env.render()

                # Explore or Exploit
                explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                if explore_p > np.random.rand():
                    # Make a random action
                    action = env.action_space.sample()
                else:
                    # Get action from Q-network
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)
                    #print (action)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)

                    if step % 10 == 0:
                        print('Episode: {}'.format(ep),
                              'Step: {:04d}'.format(t),
                              'Action: {:.4f}'.format(action),
                              'Finger position: {:.4f}'.format(env.current_finger_position),
                              'Total reward: {:.4f}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_p))
                    t = max_steps
                    rewards_list.append((ep, total_reward))

                    # Add experience to memory
                    memory.add((state, action, reward, next_state))

                    # Start new episode
                    env.reset()
                    # Take one random step to get the pole and cart moving
                    state, reward, done, _ = env.step(env.action_space.sample())

                else:
                    # Add experience to memory
                    memory.add((state, action, reward, next_state))
                    state = next_state
                    t += 1

                # Sample mini-batch from memory
                batch = memory.sample(batch_size)
                states = np.array([each[0] for each in batch])
                actions = np.array([each[1] for each in batch])
                rewards = np.array([each[2] for each in batch])
                next_states = np.array([each[3] for each in batch])

                # Train network
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                #target_Qs[episode_ends] = (0, 0, 0, 0) # Actions
                #target_Qs[episode_ends] = 0 # Actions
                target_Qs[episode_ends] = [0 for _ in range(0,7)]

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: targets,
                                               mainQN.actions_: actions})

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


test_episodes = 10
test_max_steps = env.frame_count
env.reset()
recorded_sessions = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for ep in range(1, test_episodes):
        t = 0
        recorded_frames = []
        while t < test_max_steps:
            env.render()

            # Get action from Q-network
            feed = {mainQN.inputs_: state.reshape((1, *state.shape))}
            Qs = sess.run(mainQN.output, feed_dict=feed)
            action = np.argmax(Qs)

            # Take action, get new state and reward
            next_state, reward, done, _ = env.step(action)
            finger_position = env.current_finger_position

            if done:
                t = test_max_steps
                env.reset()
                # Take one random step to get the pole and cart moving
                state, reward, done, _ = env.step(env.action_space.sample())

            else:
                state = next_state
                # Convert to Python list for Json compatibility and append
                recorded_frames.append([finger_position, np.asscalar(action)] + state.tolist()[env.key_count:])
                t += 1
        recorded_sessions.append(recorded_frames)

with open('animation.json', 'w', encoding='utf-8') as anim_file:
    json.dump(recorded_sessions, anim_file, sort_keys=True, indent=4, separators=(',', ': '))
    #json.dump(recorded_sessions, anim_file)

env.close()
