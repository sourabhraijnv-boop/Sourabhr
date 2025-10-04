import time
from collections import deque, namedtuple
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_KERAS'] = '1'
import tensorflow as tf
from pyvirtualdisplay import Display
from tensorflow import keras 
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.activations import relu, linear
import random
from collections import namedtuple
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
#Display(visible=0, size=(840, 480)).start()
tf.random.set_seed(1234)

MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEP_FOR_UPDATE = 4

env = gym.make('LunarLander-v3', render_mode = "rgb_array")
env.reset()
img = PIL.Image.fromarray(env.render())
'''
plt.imshow(img)
plt.axis('off')
plt.show()
'''
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
state_size = env.observation_space.shape
num_actions = env.action_space.n

#reset enviroment to initial state
'''
current_state, _ = env.reset()
action = 0
next_state = env.step(action)
'''
#neural network for q-function and target q-function
q_network = Sequential([
    Input(shape=state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')
])

target_q_network = Sequential([
    Input(shape=state_size),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')
])

optimizer = Adam(learning_rate=ALPHA)
@tf.function
def compute_loss(experinces, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = zip(*experinces)
    # Convert lists of tensors to proper TF tensors
    states      = tf.convert_to_tensor(states, dtype=tf.float32)
    actions     = tf.convert_to_tensor(actions, dtype=tf.int32)
    rewards     = tf.convert_to_tensor(rewards, dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
    done_vals   = tf.convert_to_tensor(done_vals, dtype=tf.float32)
    # Q(s,a)
    q_values_all = q_network(states)
    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1)
    q_values = tf.gather_nd(q_values_all, indices)

    # Q target
    next_q_values = target_q_network(next_states)
    max_next_q = tf.reduce_max(next_q_values, axis=1)
    # Use TF ops, not Python bools
    y_targets = rewards + gamma * max_next_q * (1 - done_vals)
    mse = MeanSquaredError()
    loss = mse(y_targets, q_values)
    return loss

#@tf.function to improve performance 
def agent_learn(experiences, gamma, q_network, target_q_network):
    #calculate loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    #get the gradients of the loss wrt weights
    gradients = tape.gradient(loss, q_network.trainable_variables)
    #update weight of q_network
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    target_q_network.set_weights(q_network.get_weights())

def epsilon_greedy_action(q_values, epsilon):
    """
    q_values: array-like, Q-values for each action (e.g. from policy_net(state))
    epsilon: exploration rate (0 <= epsilon <= 1)
    """
    if random.random() < epsilon:
        # Exploration
        return random.randrange(len(q_values))
    else:
        # Exploitation
        return int(np.argmax(q_values))


#learning by agent
#Deep Q learning with Experience Replay Algo with steps

start = time.time()
num_of_episodes = 2000
max_num_timesteps = 1000
total_point_history = []
num_p_av = 100  #number of total points for avrg
epsilon = 1.0

#step 1: Initialize memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)
#step 2: Initialize q_network with random weight w
#step 3: Initialize target_q_network with weight w' = w
target_q_network.set_weights(q_network.get_weights())

'''
for episode i=1 to M do 
    Recive initial observation state S[1]
    for t=1 to T do
        observe state S[t] and choose action A[t] using an e-greedy policy
        Take action A[t] in th env , recive reward R[t] and next state S[t+1]
        store experience tupple (S[t], A[t], R[t], S[t+1]) i memory buffer D
        every C(NUM_OF_STEPS_FOR_UPD) steps perform a learning update
        sample random mini-batch of experience tupple (S[j], A[j], R[j], S[j+1]) from D
        Set y[j] = R[j] if episode terminate at j+1 o/w y[j] = R[j] + gamma * maxQ_target(S[j+1], a')
        perform gradient decent step on (y[j] - Q(S[j], a[j]; w))**2 wrt Qnetwork weights w
        update weights of Qtarget with soft update
    end
end
'''
for i in range(num_of_episodes):
    # reset evv to initial state
    state, _ = env.reset()
    total_points = 0
    for t in range(max_num_timesteps):
        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn)
        #select action by e greedy policy
        action = epsilon_greedy_action(q_values, epsilon)
        #taking one step and observe
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        #storing experice in memory buffer
        memory_buffer.append(experience(state, action, reward, next_state, done))
        #updating after each c steps learn by mini batch
        if t%NUM_STEP_FOR_UPDATE == 0:
            experiences = random.sample(memory_buffer, min(len(memory_buffer), 5))
            agent_learn(experiences, GAMMA, q_network, target_q_network)
        
        state = next_state.copy()
        total_points += reward

        if done:
            break
    
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    # gradually decresing epsiolin from 1 to 0
    if epsilon >= 0.0005:
        epsilon -= 0.0005
    
    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")
    
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")