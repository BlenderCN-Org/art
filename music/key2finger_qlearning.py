import json
import re
import numpy as np
import random as rd
import tensorflow as tf


import matplotlib.pyplot as plt

from collections import deque


## Environment


def open_midi(midigram_filepath):
    ## Midigram collumns
    # 1: On (MIDI pulse units)
    # 2: Off (MIDI pulse units)
    # 3: Track number
    # 4: Channel number
    # 5: Midi pitch -> 60 -> c4 / 61 -> c#4
    # 6: Midi velocity

    midi_pulse_per_second  = 480.299
    fps = 30
    fps = 11.7 #TODO
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
        for mf_note in midi_frames[mf]:
            # Normalize pitch on the keyboard
            #mf_normalized_pitch = mf_note['pitch']-lower_pitch
            mf_normalized_pitch = mf_note['pitch'] # Disabling normalization
            if mf_normalized_pitch >= keyboard_size:
                print ("Skipping note, out of keyboard range")
                continue
            if mf_note['track'] != 1:
                print ("Skipping note, out of track")
                continue
            # Fill ontime to offtime values
            for f in range(mf_note['ontime'], mf_note['offtime']):
                music[f][mf_normalized_pitch] = 1.0

    notes = [0 for _ in range(0, keyboard_size)]
    music = [notes for _ in range(0, margin)] + music + [notes for _ in range(0, margin)]
    return music


class ActionSpace():
    def __init__(self, fingers_count):
        self.fingers_count = fingers_count

    def sample(self):
        return rd.randint(0, (5**self.fingers_count)-1)


class PianoEnv:
    def __init__(self, name, midigram_filepath):
        self.env_name = name
        self.frames = open_midi(midigram_filepath)

        self.key_count = 88
        self.fingers_count = 5
        self.frame_count = len(self.frames)
        self.training = True
        self.failed_frames_threshold = 300

        self.action_space = ActionSpace(self.fingers_count)

        print ("Generating actions")
        step_sizes = [0, 1, -1, 2, -2]
        self.finger_actions = []
        for f0 in step_sizes:
            for f1 in step_sizes:
                for f2 in step_sizes:
                    for f3 in step_sizes:
                        for f4 in step_sizes:
                            self.finger_actions.append([f0, f1, f2, f3, f4])
        #                    for f5 in step_sizes:
        #                        for f6 in step_sizes:
        #                            for f7 in step_sizes:
        #                                for f8 in step_sizes:
        #                                    for f9 in step_sizes:
        #                                        self.finger_actions.append([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9])
        print ("Actions generated")
        self.reset()



    def reset(self):
        if self.training:
            self.current_frame = rd.randint(0, self.frame_count-1)
        else:
            self.current_frame = 0
        current_key = int(self.key_count/2)
        for note_pitch, note_val in enumerate(self.frames[self.current_frame]):
            if note_val == 1.0 and current_key > note_pitch:
                current_key = note_pitch
        self.current_finger_position = []
        for n_finger in range(0, self.fingers_count):
            self.current_finger_position.append(current_key+n_finger)

        self.failed_frames = 0


    def close(self):
        pass


    def render(self):
        pass


    def step(self, action):

        step_sizes = [0, 1, -1, 2, -2, 3, -3]
        step_sizes = [0, 1, -1, 2, -2]
        # Hand 1 movements: 7
        # Hand 2 movements: 7
        # Hand 1 positions:
        # Hand 1 positions:
        finger_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # Thumb possible positions, being 4 the center (middle finger)
        finger_2 = [0, 1, 2, 3, 4, 5] # Index possible positions, being 0 the center (middle finger)
        finger_4 = [0, 1, 2, 3] # Ring possible positions, being 3 the center (middle finger)
        finger_5 = [0, 1, 2, 3, 4, 5] # Pinky possible positions, being 5 the center (middle finger)

        # 7*14*6*4*6 = 14,112
        # 7*14*6*4*6 = 14,112
        # 14.112*14.112 = 199,148,544

        # 14*6*4*6 ^ 2 = 4,064,256

        # 7*7*7*7*7 = 16,807

        # 7*5*5*5*5 = 4,375*4,375 = 19,140,625
        # 5*5*5*5*5 = 3,125*3,125 = 9,765,625



        default_reward = 0.01
        # Apply Middle finger movements
        for n_finger in range(0, self.fingers_count):
            self.current_finger_position[2] += self.finger_actions[action][2]
        if self.fingers_count > 5:
            for n_finger in range(0, self.fingers_count):
                self.current_finger_position[7] += self.finger_actions[action][7]
        for index, finger_steps in enumerate(self.finger_actions[action]):
            if index == 2 or index == 7: # If Middle fingers
                continue # movement already applyed
            else:
                self.current_finger_position[index] += finger_steps
                if index == 0:
                    # Apply constrains Left hand fingers (constrained to left Middle)
                    if self.current_finger_position[index] < self.current_finger_position[2]-5:
                        # If left Pinky is more than 5 keys away to the left, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]-5
                    elif self.current_finger_position[index] > self.current_finger_position[2]:
                        # If left Pinky is more than 0 keys away to the right, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]
                elif index == 1:
                    if self.current_finger_position[index] < self.current_finger_position[2]-3:
                        # If left Ring is more than 3 keys away to the left, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]-3
                    elif self.current_finger_position[index] > self.current_finger_position[2]:
                        # If left Ring is more than 0 keys away to the right, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]
                elif index == 3: # We skip finger 2, since it is the Left Middle finger
                    if self.current_finger_position[index] > self.current_finger_position[2]+5:
                        # If left Index is more than 5 keys away to the right, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]+5
                    elif self.current_finger_position[index] < self.current_finger_position[2]:
                        # If left Index is more than 0 keys away to the left, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]
                elif index == 4:
                    if self.current_finger_position[index] > self.current_finger_position[2]+9:
                        # If left Thumb is more than 9 keys away to the right, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]+9
                    elif self.current_finger_position[index] < self.current_finger_position[2]-4:
                        # If left Thumb is more than 4 keys away to the left, constrain
                        self.current_finger_position[index] = self.current_finger_position[2]-4
                if self.fingers_count > 5:
                    if index == 5:
                        # Apply constrains Right hand fingers (constrained to right Middle)
                        if self.current_finger_position[index] < self.current_finger_position[7]-9:
                            # If right Thumb is more than 9 keys away to the left, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]-9
                        elif self.current_finger_position[index] > self.current_finger_position[7]+4:
                            # If right Thumb is more than 4 keys away to the right, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]+4
                    elif index == 6:
                        if self.current_finger_position[index] < self.current_finger_position[7]-5:
                            # If right Index is more than 5 keys away to the left, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]-5
                        elif self.current_finger_position[index] > self.current_finger_position[7]:
                            # If right Index is more than 0 keys away to the right, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]
                    elif index == 8: # We skip finger 7, since it is the Right Middle finger
                        if self.current_finger_position[index] < self.current_finger_position[7]:
                            # If right Ring is more than 0 keys away to the left, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]
                        elif self.current_finger_position[index] > self.current_finger_position[7]+3:
                            # If right Ring is more than 3 keys away to the right, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]+3
                    elif index == 9:
                        if self.current_finger_position[index] < self.current_finger_position[7]:
                            # If right Pinky is more than 0 keys away to the left, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]
                        elif self.current_finger_position[index] > self.current_finger_position[7]+5:
                            # If right Pinky is more than 5 keys away to the right, constrain
                            self.current_finger_position[index] = self.current_finger_position[7]+5


        # One hot representation of keys with fingers on it
        one_hot = [0 for _ in range(0, self.key_count*self.fingers_count)]
        for index, finger_key in enumerate(self.current_finger_position):
            if finger_key < 0 or finger_key > self.key_count:
                continue # Skip if the finger is outside the keyboard
            try:
                one_hot[finger_key+(index*(self.key_count-1))] = 1
            except IndexError:
                print (finger_key)
                print (index)
                print (self.key_count)
                print (finger_key+(index*(self.key_count-1)))
                raise

        if 0:
            # TODO Not needed?
            key_value = 0.0
            if self.current_finger_position in self.frames[self.current_frame]:
                if key_value == 1.0:
                    reward = 0.5
                else:
                    reward = 0.01
            else:
                reward = 0.01

        done = False
        must_reset = False
        reward = 0.0

        # If no frames left, done
        if self.current_frame >= len(self.frames)-1-1:
            must_reset = True
            state = np.array(one_hot + self.frames[self.current_frame-1] + self.frames[self.current_frame-1])
        else:
            state = np.array(one_hot + self.frames[self.current_frame] + self.frames[self.current_frame + 1])

        if not must_reset:
            # If a finger falls outside the keyboard, done
            for finger_number in range(0, self.fingers_count):
                if self.current_finger_position[finger_number] < 0 or self.current_finger_position[finger_number] > self.key_count:
                    reward = 0.0001
                    must_reset = True

        if not must_reset:
            for key, key_val in enumerate(self.frames[self.current_frame]):
                if key_val == 1.0 and key not in self.current_finger_position:
                    # If a key should be pressed but is not, increase failed_frames counter
                    reward += 0.0001
                    self.failed_frames += 1
                    if self.failed_frames > self.failed_frames_threshold:
                        # If failed_frames counter > threshold, done
                        must_reset = True
                elif key_val == 1.0 and key in self.current_finger_position:
                    # If a key should be pressed and it is, reward
                    reward += 0.05

        self.current_frame += 1

        if must_reset:
            done = True

        info = {}
        return state, reward+default_reward, done, info




## Q-Learning

# Create the Cart-Pole game environment
#env = gym.make('CartPole-v0')
env = PianoEnv('CartPole-v0', 'MIDI/Hummelflug.midigram')

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
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')

            # ReLU hidden layers
            self.fc1 = tf.contrib.layers.fully_connected(self.inputs_, hidden_size)
            self.fc2 = tf.nn.dropout(tf.contrib.layers.fully_connected(self.fc1, hidden_size), self.keep_prob)
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


train_episodes = 20000         # max number of episodes to learn from
max_steps = env.frame_count    # max steps in an episode
gamma = 0.9                    # future reward discount

# Exploration parameters
explore_start = 0.99            # exploration probability at start
explore_stop = 0.001            # minimum exploration probability
decay_rate = 0.000001           # exponential decay rate for exploration prob

# Network parameters
hidden_size = 128              # number of units in each Q-network hidden layer
learning_rate = 0.00001        # Q-network learning rate

# Memory parameters
memory_size = 1000000          # memory capacity
batch_size = 128               # experience mini-batch size
pretrain_length = batch_size   # number experiences to pretrain the memory

do_training = False

tf.reset_default_graph()
state_size = (env.key_count * 2) + (env.key_count * env.fingers_count)
action_size = 5 ** env.fingers_count
print ("State Size: {}, Action Size: {}".format(state_size, action_size))
mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, state_size=state_size, action_size=action_size)

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
                    feed = {mainQN.inputs_: state.reshape((1, *state.shape)), mainQN.keep_prob: 1.0}
                    Qs = sess.run(mainQN.output, feed_dict=feed)
                    action = np.argmax(Qs)
                    #print (action)

                # Take action, get new state and reward
                next_state, reward, done, _ = env.step(action)

                total_reward += reward

                if done:
                    # the episode ends so no next state
                    next_state = np.zeros(state.shape)

                    if step % 100 == 0:
                        saver.save(sess, "checkpoints/cartpole.ckpt")

                    if step % 10 == 0:
                        print('Episode: {}'.format(ep),
                              'Step: {:04d}'.format(t),
                              'Action: {:.4f}'.format(action),
                              #'Finger position: {:.4f}'.format(env.current_finger_position),
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
                target_Qs = sess.run(mainQN.output, feed_dict={mainQN.inputs_: next_states, mainQN.keep_prob: 0.7})

                # Set target_Qs to 0 for states where episode ends
                episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                #target_Qs[episode_ends] = (0, 0, 0, 0) # Actions
                #target_Qs[episode_ends] = 0 # Actions
                target_Qs[episode_ends] = [0 for _ in range(0,action_size)]

                targets = rewards + gamma * np.max(target_Qs, axis=1)

                loss, _ = sess.run([mainQN.loss, mainQN.opt],
                                    feed_dict={mainQN.inputs_: states,
                                               mainQN.targetQs_: targets,
                                               mainQN.actions_: actions,
                                               mainQN.keep_prob: 1.0})

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
test_max_steps = env.frame_count
env.reset()
env.training = False
recorded_sessions = []
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    for ep in range(1, test_episodes):
        t = 0
        recorded_frames = []
        while t < test_max_steps:
            env.render()

            # Get action from Q-network
            feed = {mainQN.inputs_: state.reshape((1, *state.shape)), mainQN.keep_prob: 0.7}
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

env.close()
