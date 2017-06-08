import json
import re
import numpy as np
import random as rd
import tensorflow as tf

import matplotlib.pyplot as plt

from collections import deque


class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4,
                 action_size=2, hidden_size=10,
                 name='QNetwork', optimize=True):
        # state inputs to the Q-network
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            #self.inputs_ = tf.placeholder(tf.float32, [state_size], name='inputs')

            # One hot encode the actions to later choose the Q-value for the action
            with tf.variable_scope('actions'):
                self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
                one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')


            with tf.variable_scope('reshape'):
                self.inputs2d = tf.reshape(self.inputs_, [self.batch_size, 12, 88, 1]) # batch_size * 88*12
                tf.summary.image("input", self.inputs2d)
                # 1D "image" 88 pixels width, and 12 channels
                self.inputs2d = tf.transpose(self.inputs2d, perm=[0, 2, 3, 1])
                #print (self.inputs2d.shape)
                #lala()

            with tf.variable_scope('convolutions'):
                filters_count = 32
                kernel_size = [4, 1] # Width and Height
                strides = [2, 1] # Height and Width = 88 -> 43x32
                padding = "same"
                self.cv1 = tf.layers.conv2d(self.inputs2d, filters_count, kernel_size, strides, padding, activation=tf.nn.relu6)
                print ("CV1", self.cv1.shape) #88 1 32
                reshaped = tf.reshape(self.cv1, [self.batch_size, int(self.cv1.shape[1]), int(self.cv1.shape[3]), 1])
                reshaped = tf.transpose(reshaped, perm=[0, 2, 1, 3])
                tf.summary.image("cv1", reshaped)

                filters_count = 64
                kernel_size = [4, 1] # Width and Height
                strides = [2, 1] # Height and Width = 44 -> 20x64
                self.cv2 = tf.layers.conv2d(self.cv1, filters_count, kernel_size, strides, padding, activation=tf.nn.relu6)
                print ("CV2", self.cv2.shape) #44 1 64
                reshaped = tf.reshape(self.cv2, [self.batch_size, int(self.cv2.shape[1]), int(self.cv2.shape[3]), 1])
                reshaped = tf.transpose(reshaped, perm=[0, 2, 1, 3])
                tf.summary.image("cv2", reshaped)

                filters_count = 128
                kernel_size = [4, 1] # Width and Height
                strides = [2, 1] # Height and Width = 22 -> 9x128
                self.cv3 = tf.layers.conv2d(self.cv2, filters_count, kernel_size, strides, padding, activation=tf.nn.relu6)
                print ("CV3", self.cv3.shape) #22 1 128
                reshaped = tf.reshape(self.cv3, [self.batch_size, int(self.cv3.shape[1]), int(self.cv3.shape[3]), 1])
                reshaped = tf.transpose(reshaped, perm=[0, 2, 1, 3])
                tf.summary.image("cv3", reshaped)

            self.fc1 = tf.contrib.layers.flatten(self.cv3)

            with tf.variable_scope('fully_connected_layers'):
                self.fc2 = tf.nn.dropout(tf.contrib.layers.fully_connected(self.fc1, hidden_size), self.keep_prob)
                self.fc3 = tf.contrib.layers.fully_connected(self.fc2, hidden_size)

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(self.fc3, action_size,
                                                            activation_fn=None)

            tf.summary.histogram('output', self.output)
            ### Train with loss (targetQ - Q)^2
            # output has length of possible actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded action.
            # Example: [1.2, 3.4, 9.3] x [1, 0, 0] = [1.2, 0, 0] = 1.2
            with tf.variable_scope('selected_Q'):
                self.Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)

            tf.summary.histogram('Q', self.Q)
            tf.summary.histogram('target_Q', self.targetQs_)

            if optimize:
                with tf.variable_scope('optimize_loss'):
                    # According to doi:10.1038/nature14236 clipping
                    # Because the absolute value loss function x
                    # has a derivative of -1 for all negative values of x
                    # and a derivative of 1 for all positive values of x
                    # , clipping the squared error to be between -1 and 1 cor-
                    # responds to using an absolute value loss function for
                    # errors outside of the (-1,1) interval.
                    # This form of error clipping further improved the stability of the algorithm.

                    self.loss = tf.reduce_mean(self.clipped_error(self.targetQs_ - self.Q))
                    tf.summary.scalar('loss', self.loss)
                    #self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                    self.opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.95).minimize(self.loss)

            self.merged = tf.summary.merge_all()


        def clipped_error(self, x):
            # Huber Loss
            return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5) # condition, true, false



class Memory():
    def __init__(self, max_size = 1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample_(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]

    def sample(self, batch_size):
        negative_rewards=[]
        zero_rewards=[]
        positive_rewards=[]
        for buffer_item in self.buffer:
            if buffer_item[2] == -1:
                negative_rewards.append(buffer_item) # Select negative_rewards
            elif buffer_item[2] == 0:
                zero_rewards.append(buffer_item) # Select zero_rewards
            elif buffer_item[2] == 1:
                positive_rewards.append(buffer_item) # Select positive_rewards

        negative_length=int(batch_size*0.3)
        zero_length=int(batch_size*0.3)
        positive_length=int(batch_size*0.3)

        sum_length = positive_length+zero_length+negative_length
        if sum_length < batch_size:
            positive_length += batch_size-sum_length

        spare = 0
        if positive_length > len(positive_rewards):
            spare += positive_length - len(positive_rewards)
            positive_length = len(positive_rewards)
        if zero_length > len(zero_rewards):
            spare += zero_length - len(zero_rewards)
            zero_length = len(zero_rewards)
        negative_length += spare

        #print (len(negative_rewards), len(zero_rewards), len(positive_rewards), negative_length, zero_length, positive_length)

        idxn = []
        idxz = []
        idxp = []

        if negative_length > 0:
            idxn = np.random.choice(np.arange(len(negative_rewards)),
                                   size=negative_length,
                                   replace=False)
        if zero_length > 0:
            idxz = np.random.choice(np.arange(len(zero_rewards)),
                                   size=zero_length,
                                   replace=False)
        if positive_length > 0:
            idxp = np.random.choice(np.arange(len(positive_rewards)),
                                   size=positive_length,
                                   replace=False)

        return [negative_rewards[ii] for ii in idxn]+[zero_rewards[ii] for ii in idxz]+[positive_rewards[ii] for ii in idxp]


#def save_json(json_path, data):
#    with open(json_path, 'w', encoding='utf-8') as anim_file:
#        json.dump(data, anim_file, sort_keys=True, indent=4, separators=(',', ': '))

def training():
    train_episodes = 10000         # max number of episodes to learn from
    #training_steps = 10
    max_steps = 100                # max steps in an episode
    gamma = 0.99                   # future reward discount
    update_frequency = 10000

    # Exploration parameters
    explore_start = 0.99           # exploration probability at start
    explore_stop = 0.1             # minimum exploration probability
    #decay_rate = 0.000001          # exponential decay rate for exploration prob

    # Network parameters
    hidden_size = 512              # number of units in each Q-network hidden layer
    learning_rate = 0.00025        # Q-network learning rate

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
    mainQN = QNetwork(name='main', hidden_size=hidden_size, learning_rate=learning_rate, state_size=state_size, action_size=action_size, optimize=False)
    copyQN = QNetwork(name='copy', hidden_size=hidden_size, learning_rate=learning_rate, state_size=state_size, action_size=action_size)

    # Copy Op
    def get_var(varname):
        ret = [v for v in tf.global_variables() if v.name == varname]
        if len(ret) == 0:
            print ("\"{}\" not found".format(varname))
            return None
        return ret[0]

    vars2copy = []
    vars2save = {}
    for vvar in tf.global_variables():
        if vvar.name.startswith('main/'):
            # Copy the following vars
            vars2copy.append(vvar.name[5:])
            # Save the following vars
            if get_var(vvar.name) is not None:
                vars2save[vvar.name] = get_var(vvar.name)

    copying_cm = []
    with tf.variable_scope('copy_parameters_cm'):
        for vvar in vars2copy:
            fromvar = get_var('copy/{}'.format(vvar))
            tovar = get_var('main/{}'.format(vvar))
            if fromvar is not None and tovar is not None:
                    copying_cm.append(tovar.assign(fromvar))

    copying_mc = []
    with tf.variable_scope('copy_parameters_mc'):
        for vvar in vars2copy:
            fromvar = get_var('main/{}'.format(vvar))
            tovar = get_var('copy/{}'.format(vvar))
            if fromvar is not None and tovar is not None:
                    copying_mc.append(tovar.assign(fromvar))
    #

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
        if 1.0 > np.random.rand():
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

    saver = tf.train.Saver(vars2save)

    loss = 0.0
    if do_training:
        print ("Training")
        # Now train with experiences
        rewards_list = []
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run(copying_cm) # Perform copy parameters from copy to main

            file_writer = tf.summary.FileWriter('./logs/1', sess.graph)
            # Load to Main
            if False:
                saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
                sess.run(copying_mc) # Perform copy parameters from copy to main

            step = 0
            global_step = 0
            for ep in range(1, train_episodes):
                total_reward = 0
                t = 0
                positive_rewards = 0
                while t < max_steps:
                    step += 1
                    # Uncomment this next line to watch the training
                    if ep % 10 == 0:
                        env.render()

                    # Explore or Exploit
                    #explore_p = explore_stop + (explore_start - explore_stop)*np.exp(-decay_rate*step)
                    explore_p = np.clip(1.0-np.sin(global_step*0.0001), explore_stop, explore_start)
                    if explore_p > np.random.rand():
                        # Make a random action
                        #if state is None or 1.0 > np.random.rand():
                        action = env.action_space.sample() # take a random action
                        #else:
                        #    action = get_action(state) # take a predefined action
                    else:
                        # Get action from Q copy
                        feed = {
                            copyQN.inputs_: state.reshape((1, *state.shape)),
                            copyQN.batch_size: 1,
                            copyQN.keep_prob: 1.0,
                            # Dummy values, not used, only to satisfy the Graph
                            mainQN.inputs_: state.reshape((1, *state.shape)),
                            mainQN.batch_size: 1,
                            mainQN.keep_prob: 1.0
                        }
                        Qs = sess.run(copyQN.output, feed_dict=feed)
                        action = np.argmax(Qs)

                    # Take action, get new state and reward
                    next_state, reward, done, _ = env.step(action)

                    total_reward += reward
                    if reward > 0:
                        positive_rewards += 1

                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros(state.shape)

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

                    # Sample mini-batch from memory
                    batch = memory.sample(batch_size)
                    states = np.array([each[0] for each in batch])
                    actions = np.array([each[1] for each in batch])
                    rewards = np.array([each[2] for each in batch])
                    next_states = np.array([each[3] for each in batch])

                    # Get main Q^ #
                    # Executes batch_size actions, and caches the output
                    # of the Neural Network. output = Q
                    feed_dict = {
                        mainQN.inputs_: next_states,
                        mainQN.batch_size: batch_size,
                        mainQN.keep_prob: 1.0
                    }
                    target_Qs = sess.run(mainQN.output, feed_dict=feed_dict)

                    data = sess.run(mainQN.inputs2d, feed_dict=feed_dict)

                    # Set target_Qs to 0 for states where episode ends
                    # Ending episode + 1 should have zero Q (all the reward
                    # is "stored" on the current reward)
                    episode_ends = (next_states == np.zeros(states[0].shape)).all(axis=1)
                    target_Qs[episode_ends] = [0 for _ in range(0,action_size)]

                    # Updates the already generated Q according to the reward
                    # like if the generated Q where real (?)
                    targets = rewards + gamma * np.max(target_Qs, axis=1)

                    # Train copy Network #
                    # Force the network to output the new Q
                    # given the same state and action as before
                    feed_dict = {
                        copyQN.inputs_: states,
                        copyQN.targetQs_: targets,
                        copyQN.actions_: actions,
                        copyQN.batch_size: batch_size,
                        copyQN.keep_prob: 7.0,
                        # Dummy values, not used, only to satisfy the Graph
                        mainQN.inputs_: states,
                        mainQN.actions_: actions,
                        mainQN.targetQs_: targets,
                        mainQN.batch_size: batch_size,
                        mainQN.keep_prob: 7.0,
                    }
                    summary, loss, _ = sess.run([copyQN.merged, copyQN.loss, copyQN.opt],
                                        feed_dict=feed_dict)

                    if global_step % 100 == 0:
                        file_writer.add_summary(summary, global_step)

                    if global_step % update_frequency == 0:
                        sess.run(copying_cm) # Perform copy parameters from copy to main

                    if ep % 10 == 0:
                        saver.save(sess, "checkpoints/cartpole.ckpt")

                    if t >= max_steps:
                        print('Episode: {}'.format(ep),
                              #'Step: {:04d}'.format(t),
                              #'Action: {:.4f}'.format(action),
                              #'Finger position: {:.4f}'.format(env.current_finger_position),
                              'Memory Size: {}'.format(len(memory.buffer)),
                              'Positive rewards: {}'.format(positive_rewards),
                              'Total reward: {}'.format(total_reward),
                              #'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_p))
                        env.reset()

                    global_step += 1

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

def testing():
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

                if 0.05 > np.random.rand():
                    state, reward, done, _ = env.step(env.action_space.sample())
                else:
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
