import random
from collections import namedtuple

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import numpy as np

import cv2


class GameNetwork(nn.Module):
    OUTPUT_AUGMENT = 5

    WorldState = namedtuple("WorldState", "actions observations world_image")

    def __init__(self,
                 actions_size,
                 actions_history_size,
                 possible_actions,
                 observation_size,
                 image_height=0,
                 image_width=0):
        super(GameNetwork, self).__init__()

        actions_input_size = actions_size * actions_history_size

        self.actions_to_hidden = nn.Linear(actions_input_size, actions_history_size)
        hidden_input_size = actions_history_size

        self.observation_to_hidden = nn.Linear(observation_size, observation_size * actions_input_size)
        hidden_input_size += (observation_size * actions_input_size)

        # Image
        #self.shrink_1_4 = nn.Conv2d(4, 4, kernel_size=6, stride=2, padding=1)
        #self.batch_norm_4chan_1 = nn.BatchNorm2d(4)

        #self.shrink_1_2 = nn.Conv2d(4, 4, kernel_size=4, stride=2, padding=1)
        #self.batch_norm_4chan_2 = nn.BatchNorm2d(4)

        self.conv_layer_a = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.batch_norm_16chan = nn.BatchNorm2d(16)

        self.conv_layer_b = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.batch_norm_32chan_1 = nn.BatchNorm2d(32)

        self.conv_layer_c = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.batch_norm_32chan_2 = nn.BatchNorm2d(32)
        hidden_input_size += 896

        self.hidden_to_out_hidden = nn.Linear(hidden_input_size, possible_actions * self.OUTPUT_AUGMENT)

        self.output_layer = nn.Linear(possible_actions * self.OUTPUT_AUGMENT, possible_actions)

        self.optimizer = None
        self.loss_fn = None

    def process_state(self, actions_history, memory_observations, world_image):
        world_image = cv2.resize(world_image, (80, 60))

        b, g, r = cv2.split(world_image)
        grey = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)

        imgarray = np.array([b, g, r, grey]).astype(np.float32)

        return self.WorldState(np.ndarray.flatten(np.array(actions_history)).astype(dtype=np.float32),
                               np.array(memory_observations, dtype=np.float32),
                               imgarray)

    def forward(self, world_states):
        # Tensors from inputs
        actions, observations, world_images = map(np.array, zip(*world_states))
        actions = Variable(torch.from_numpy(actions).cuda(), requires_grad=True).cuda()
        observations = Variable(torch.from_numpy(observations).cuda(), requires_grad=True).cuda()
        world_images = Variable(torch.from_numpy(world_images).cuda(), requires_grad=True).cuda()

        # Conv Layers
        #print("World shape",world_images.shape)
        #conv_1_4 = F.relu(self.batch_norm_4chan_1(self.shrink_1_4(world_images)))
        #conv_1_2 = F.relu(self.batch_norm_4chan_2(self.shrink_1_2(conv_1_4)))
        #print("Shrink size",conv_1_2.shape)

        conv_output = F.relu(self.batch_norm_16chan(self.conv_layer_a(world_images)))
        conv_output = F.relu(self.batch_norm_32chan_1(self.conv_layer_b(conv_output)))
        conv_output = F.relu(self.batch_norm_32chan_2(self.conv_layer_c(conv_output)))

        # Linear Layers
        actions_hidden = self.actions_to_hidden(actions)
        observations_hidden = self.observation_to_hidden(observations)

        #print("Conv out =", conv_output.shape)

        hidden = torch.cat((actions_hidden,
                            observations_hidden,
                            conv_output.view(conv_output.size(0), -1)), dim=1)

        out_hidden = self.hidden_to_out_hidden(hidden)
        return self.output_layer(out_hidden)

    def get_best_action(self, world_state):
        return self([world_state]).cuda().max(1)[1].view(1, 1)


class GameNetworkTrainer:
    Replay = namedtuple("Replay", "state action reward next_state")
    GAMMA = 0.983

    def __init__(self, actions_size,
                 actions_history_size,
                 possible_actions,
                 observation_size,
                 replays_memory=550,
                 sample_batch_size=131):
        self.policy_network = GameNetwork(actions_size, actions_history_size, possible_actions, observation_size)
        self.target_network = GameNetwork(actions_size, actions_history_size, possible_actions, observation_size)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        if torch.cuda.is_available():
            self.policy_network.cuda()
            self.target_network.cuda()

        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=1e-5)
        self.loss_fn = torch.nn.SmoothL1Loss(reduction='mean')

        self.replays = [None] * replays_memory
        self.replays_index = 0
        self.sample_batch_size = sample_batch_size

    def add_replay(self, old_state, action, reward, new_state):
        replay = self.Replay(old_state, action, reward, new_state)
        self.replays[self.replays_index] = replay

        self.replays_index = (self.replays_index + 1) % len(self.replays)

    def train(self):
        batch = random.sample(self.replays, self.sample_batch_size)
        old_state_batch, action_batch, reward_batch, next_state_batch = map(np.array, zip(*batch))

        action_batch = torch.from_numpy(action_batch).cuda().view(-1, 1)

        predicted_values = self.policy_network(old_state_batch).gather(1, action_batch)
        next_predicted_values = self.target_network(next_state_batch).max(1)[0]

        expected_values = (next_predicted_values * self.GAMMA).double() + torch.from_numpy(reward_batch).cuda().double()

        loss = self.loss_fn(predicted_values.double().cuda(), expected_values)
        self.optimizer.zero_grad()
        loss.backward()
        for p in self.policy_network.parameters():
            p.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def get_policy_network(self):
        return self.policy_network

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())