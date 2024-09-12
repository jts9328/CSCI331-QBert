import gymnasium as gym
import math
import random
import matplotlib
from collections import namedtuple, deque
from itertools import count
import pdb
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
from tqdm.notebook import tqdm_notebook
import matplotlib.pyplot as plt
import stable_baselines3

env = gym.make("QbertNoFrameskip-v4")
#env = gym.make("CartPole-v1")

# preprocessing of env from Medium article: https://medium.com/nerd-for-tech/reinforcement-learning-deep-q-learning-with-atari-games-63f5242440b1
# sets noop max to 30, a frame skip of 4 frames, makes the screen 84x84, sets color to grayscale, and clips reward to [-1, 0, +1]
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
env = gym.wrappers.FrameStack(env, 4)
# env = stable_baselines3.common.atari_wrappers.ClipRewardEnv(env)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#action selection with online network
# best_future_action = np.argmax(model.predict(np.expand_dims(new_state, axis=0)))
#action evaluation with target network
# target = reward + discount_factor * target_model.predict(np.expand_dims(new_state, axis=0))[0][best_future_action]

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN_new(nn.Module):
    def __init__(self, env_type):
        self.env_type = env_type
        self.env = gym.make(env)
    
    def show(self, image):
        pass

    def getIm(self):
        pass

    def reset(self):
        return self.env.reset()

    def run(self):
        pass

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        #self.layer1 = nn.Linear(n_observations, 128)
        #self.layer2 = nn.Linear(128, 128)
        #self.layer3 = nn.Linear(128, n_actions)
        self.layer1 = nn.Conv2d(n_observations, 84, kernel_size=8, stride=4)   # research suggests: k=8 s=4
        self.layer2 = nn.Conv2d(84, 84, kernel_size=4, stride=2)              # k=4 s=2
        self.layer3 = nn.Conv2d(84, n_actions, kernel_size=3, stride=1)        # k=3 s=1
        self.lin1 = nn.Linear(294, 84)
        self.lin2 = nn.Linear(84, n_actions)
        self.flat = nn.Flatten()

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        #x = self.flat(x)
        #x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        #pdb.set_trace()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        x = self.flat(x)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
state, info = env.reset()
n_observations = env.observation_space.shape[0]    # 100800

parser = argparse.ArgumentParser(
    prog='q-learning',
    description='learns to play a game'
)

parser.add_argument('-s', '--save', default="qbert_wrapped.pytorch", help="file to save model to", type=str)
parser.add_argument('-l', '--load', default="qbert_wrapped.pytorch", help="file to load model from", type=str)

args = parser.parse_args()

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
if args.load is not None:
    print (f"loading {args.load}")
    policy_net.load_state_dict(torch.load(args.save)) ############################ args.load?
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #return policy_net(state).max(1)[1].view(0, 0) #############################
            action = policy_net(state).max(1)[1].item()  # Get the action index with .item()
            return torch.tensor([[action]], device=device, dtype=torch.long)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
episode_scores = []


def plot_durations(show_result=False):
    plt.figure(1)
    # durations_t = torch.tensor(episode_durations, dtype=torch.float)
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    # plt.ylabel('Duration')
    plt.ylabel('Score')
    # plt.plot(durations_t.numpy())
    plt.plot(scores_t.numpy())
    # Take 100 episode averages and plot them too
    # if len(durations_t) >= 100:
    if len(scores_t) >= 100:
        # means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def run_model(count = 10000):
    """You should probably not modify this, other than
    to load qbert.
    """
    env = gym.make("QbertNoFrameskip-v4", render_mode="human")
    # preprocessing of env from Medium article: https://medium.com/nerd-for-tech/reinforcement-learning-deep-q-learning-with-atari-games-63f5242440b1
    # sets noop max to 30, a frame skip of 4 frames, makes the screen 84x84, sets color to grayscale, and clips reward to [-1, 0, +1]
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                        screen_size=84, terminal_on_life_loss=False, 
                                        grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    # env = stable_baselines3.common.atari_wrappers.ClipRewardEnv(env)
    #env = gym.make("CartPole-v1", render_mode="human")

    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in range(count):
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            state = None
        else:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        env.render()
        if done:
            break

def train_model():
    """ You may want to modify this method: for instance,
    you might want to skip frames during training."""
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 10

    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        game_score = 0
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            buffer = deque(maxlen=4)
            reward_sum = 0.0
            done = None
            for _ in range(4):
                observation, reward, terminated, truncated, _ = env.step(action.item())
                buffer.append(observation)
                reward_sum += reward
                if done:
                    break
            observation = np.max(np.stack(buffer), axis=0)
            reward = torch.tensor([reward_sum], device=device)
            print(reward_sum)
            game_score += reward_sum
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                # episode_durations.append(t + 1)
                episode_scores.append(game_score)
                plot_durations()
                break
# train_model()
torch.save(policy_net.state_dict(), args.save)

# print('Complete')
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
run_model(10000)


