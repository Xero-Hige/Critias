import random
from collections import namedtuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MushaNetwork(nn.Module):
    OUTPUT_AUGMENT = 10

    WorldState = namedtuple("WorldState", "world_image")

    def __init__(self,
                 possible_actions,
                 debug=False):
        super(MushaNetwork, self).__init__()

        self.debug = debug
        self.has_cuda = torch.cuda.is_available()

        self.max_pool_shrink_1_2 = torch.nn.MaxPool2d(2, 2)
        self.max_pool_conv_a = torch.nn.MaxPool2d((2, 2), (1, 2))
        self.max_pool_conv_b = torch.nn.MaxPool2d(4, (2, 1))
        self.max_pool_conv_c = torch.nn.MaxPool2d((2, 3), 2)

        # Image
        self.conv_input_a = nn.Conv2d(4, 8, kernel_size=4)
        self.conv_input_a_bn = nn.BatchNorm2d(8)

        self.conv_input_b = nn.Conv2d(8, 16, kernel_size=5)
        self.conv_input_b_bn = nn.BatchNorm2d(16)

        self.conv_layer_a = nn.Conv2d(16, 32, kernel_size=(2, 3))
        self.conv_layer_a_bn = nn.BatchNorm2d(32)

        self.conv_layer_b = nn.Conv2d(32, 128, kernel_size=(3, 2))
        self.conv_layer_b_bn = nn.BatchNorm2d(128)

        self.conv_layer_c = nn.Conv2d(128, 256, kernel_size=5)
        self.conv_layer_c_bn = nn.BatchNorm2d(256)

        hidden_input_size = 32256
        self.hidden_layer = nn.Linear(hidden_input_size, possible_actions * self.OUTPUT_AUGMENT)
        self.output_layer = nn.Linear(possible_actions * self.OUTPUT_AUGMENT, possible_actions)

        self.optimizer = None
        self.loss_fn = None

    def set_debug(self,debug):
        self.debug = debug

    def process_state(self, world_image):
        b, g, r = cv2.split(world_image)
        grey = cv2.cvtColor(world_image, cv2.COLOR_BGR2GRAY)

        imgarray = np.array([b, g, r, grey]).astype(np.float32)

        return self.WorldState(imgarray)

    def forward(self, world_states):
        # Tensors from inputs
        world_images = np.array(list(zip(*world_states)))[0]
        world_images = Variable(torch.from_numpy(world_images), requires_grad=True)
        if self.has_cuda:
            world_images = world_images.cuda()

        conv_output = F.relu(self.max_pool_shrink_1_2(self.conv_input_a_bn(self.conv_input_a(world_images))))

        if self.debug:
            channels = conv_output[0].detach().numpy()

            channels = [np.concatenate(channels[i:i + 2]) for i in range(8 // 2)]
            channels = np.concatenate(channels, axis=1)
            cv2.imshow("First 2dconv", channels)

        conv_output = F.relu(self.max_pool_shrink_1_2(self.conv_input_b_bn(self.conv_input_b(conv_output))))

        if self.debug:
            channels = conv_output[0].detach().numpy()
            channels = [np.concatenate(channels[i:i + 4]) for i in range(16 // 4)]
            channels = np.concatenate(channels, axis=1)
            cv2.imshow("Second 2dconv", channels)

        conv_output = F.relu(self.max_pool_conv_a(self.conv_layer_a_bn(self.conv_layer_a(conv_output))))

        if self.debug:
            channels = conv_output[0].detach().numpy()
            channels = [np.concatenate(channels[i:i + 4]) for i in range(32 // 4)]
            channels = np.concatenate(channels, axis=1)
            cv2.imshow("Trird conv", channels)

        conv_output = F.relu(self.max_pool_conv_b(self.conv_layer_b_bn(self.conv_layer_b(conv_output))))

        if self.debug:
            channels = conv_output[0].detach().numpy()
            channels = [np.concatenate(channels[i:i + 16]) for i in range(128 // 16)]
            channels = np.concatenate(channels, axis=1)
            cv2.imshow("Fourth conv", channels)

        conv_output = F.relu(self.max_pool_conv_c(self.conv_layer_c_bn(self.conv_layer_c(conv_output))))

        if self.debug:
            channels = conv_output[0].detach().numpy()
            channels = [np.concatenate(channels[i:i + 16]) for i in range(256 // 16)]
            channels = np.concatenate(channels, axis=1)
            cv2.imshow("Fifth conv", channels)

        if self.debug:
            cv2.waitKey(1)

        hidden = F.relu(self.hidden_layer(conv_output.view(conv_output.size(0), -1)))
        return self.output_layer(hidden)

    def get_best_action(self, world_state):
        if self.has_cuda:
            return self([world_state]).cuda().max(1)[1].view(1, 1)
        return self([world_state]).max(1)[1].view(1, 1)


class GameNetworkTrainer:
    Replay = namedtuple("Replay", "state action reward next_state")
    GAMMA = 0.983

    def __init__(self,
                 possible_actions,
                 observation_size,
                 replays_memory=550,
                 sample_batch_size=131,
                 save_path=None):

        self.has_cuda = torch.cuda.is_available()

        self.policy_network = MushaNetwork(possible_actions)
        self.target_network = MushaNetwork(possible_actions)

        if save_path:
            self.policy_network.load_state_dict(torch.load(save_path, map_location='cpu'))
            self.policy_network.eval()

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

        action_batch = torch.from_numpy(action_batch).view(-1, 1)
        if self.has_cuda:
            action_batch = action_batch.cuda()

        predicted_values = self.policy_network(old_state_batch).gather(1, action_batch)
        next_predicted_values = self.target_network(next_state_batch).max(1)[0]

        if self.has_cuda:
            expected_values = (next_predicted_values * self.GAMMA).double() + torch.from_numpy(
                reward_batch).double().cuda()
        else:
            expected_values = (next_predicted_values * self.GAMMA).double() + torch.from_numpy(reward_batch).double()

        if self.has_cuda:
            predicted_values = predicted_values.cuda()

        loss = self.loss_fn(predicted_values.double().reshape(-1), expected_values.reshape(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_policy_network(self):
        return self.policy_network

    def update_target(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def store_policy(self, path):
        torch.save(self.policy_network.state_dict(), path)
