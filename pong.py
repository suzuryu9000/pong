"""
Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym.
https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
"""
import numpy as np
import pickle
import gym

# hyperparameters
H = 200                 # Number of hidden layer neurons
# How many games we play before updating the weights of our network.
batch_size = 10
# The rate at which we learn from our results to compute the new weights. A higher rate means
learning_rate = 1e-4
# 	we react more to results and a lower rate means we don’t react as strongly to each result.

# The discount factor we use to discount the effect of old actions on the final result.
gamma = 0.99
decay_rate = 0.99       # Parameter used in RMSProp
resume = True           # Resume from previous checkpoint?
render = True           # Display game window

# model initialization
D = 80 * 80  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('weights.pkl', 'rb'))
else:
    model = {
        # Neuron (row) i, for pizel j
        'W1': np.random.randn(H, D) / np.sqrt(D),
        # the weights we place on the activation of neuron i in the hidden layer
        'W2': np.random.randn(H) / np.sqrt(H)
    } 	# We divide by the square root of the number of the dimension size to normalize our weights.

# Store intermediate values for backpropagation
# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v)
                 for k, v in model.items()}  # rmsprop memory


def sigmoid(x):
    """
    Sigmoid "squashing" function to interval [0,1]
    # https://en.wikipedia.org/wiki/Sigmoid_function
    """
    # TODO
    pass


def preprocess(I):
    """
    Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
    """
    I = I[35:195]    # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1    # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """
    Take 1D float array of rewards and compute discounted reward
    r: set of rewards for a bunch of timesteps
    Weight the most immediate rewards higher than the later rewards, exponentially.
    As time steps go forward, the rewards are exponentially decreased in importance.
    Optimizing for short term. Pong is fast paced. Not much strategy involved.

    # https://github.com/hunkim/ReinforcementZeroToAll/issues/1
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    """
    Forward pass
    Will be able to detect various game scenarios (e.g. the ball is in the top, and our paddle is in the middle)
    """
    # TODO
    pass


def policy_backward(eph, epdlogp):
    """
    Backward pass.
    Recursively computing derivatives, with respect to weights, at each layer.
    Why Chain rule? We are taking the values at each layer, and using those
    values to compute the next set of partial derivatives (gradients).

    eph: array of intermediate hidden states
    epdlogp: modulates the gradient with Advantage

    # https://www.youtube.com/watch?v=q555kfIFUCM
    # https://www.youtube.com/watch?v=Ilg3gGewQ5U&vl=en
    """
    dW2 = np.dot(eph.T, epdlogp).ravel()  # derivative with respect to weight 2
    dh = np.outer(epdlogp, model['W2'])  # derivative of hidden state
    dh[eph <= 0] = 0                     # backprop relu
    dW1 = np.dot(dh.T, epx)              # derivative with respect to weight 1
    # return both derivatives to update weights
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None   # We want our policy network to detect motion.
# Used in computing the difference frame

episode_observations = []
episode_hidden_layer_values = []
episode_gradient_log_ps = []
episode_rewards = []

running_reward = None
reward_sum = 0
episode_number = 0

# Begin training
while True:
    if render:
        env.render()

    # preprocess the observation, set input to network to be difference image

    # forward the policy network and sample an action from the returned probability

    # record various intermediates (needed later for backprop)

    # gradient that encourages the action that was taken to be taken
    # see http://cs231n.github.io/neural-networks-2/#losses if confused
    episode_gradient_log_ps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)  # take action
    reward_sum += reward

    # record reward (has to be done after we call step() to get reward for previous action)
    episode_rewards.append(reward)

    if done:  # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(episode_observations)
        eph = np.vstack(episode_hidden_layer_values)
        epdlogp = np.vstack(episode_gradient_log_ps)
        epr = np.vstack(episode_rewards)

        episode_observations = []
        episode_hidden_layer_values = []
        episode_gradient_log_ps = []
        episode_rewards = []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * \
                    rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / \
                    (np.sqrt(rmsprop_cache[k]) + 1e-5)
                # reset batch gradient buffer
                grad_buffer[k] = np.zeros_like(v)

        running_reward = reward_sum if running_reward is None else running_reward * \
            0.99 + reward_sum * 0.01
        print('Episode %d reward total was %f. running mean: %f' %
              (episode_number, reward_sum, running_reward))

        if episode_number % 100 == 0:
            print('Saving model... ')
            pickle.dump(model, open('weights.pkl', 'wb'))

        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
