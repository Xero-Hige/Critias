from __future__ import print_function

from collections import namedtuple

import retro
import numpy as np
import cv2
import pickle
import random

from network_trainer import GameNetworkTrainer

# random.seed(99)

ENVIRONMENT = retro.make("SonicAndKnuckles3-Genesis", "AngelIslandZone.Act1")

State = namedtuple("State", "world x_position y_position rings lives score done reward s_x, s_y, end_x")

# ACTION: [0, 0, 0, 0, UP, DOWN, LEFT, RIGHT, JUMP, 0, 0, 0]

ACTIONS = (
    ("UP", [1, 0, 0, 0, 0]),
    ("DOWN", [0, 1, 0, 0, 0]),
    ("LEFT", [0, 0, 1, 0, 0]),
    ("RIGHT", [0, 0, 0, 1, 0]),
    ("JUMP", [0, 0, 0, 0, 1]),
    ("JUMP_UP", [1, 0, 0, 0, 1]),
    ("CHARGE", [0, 1, 0, 0, 1]),
    ("JUMP_LEFT", [0, 0, 1, 0, 1]),
    ("RIGHT_JUMP", [0, 0, 0, 1, 1]),
    ("DOWN_LEFT", [0, 1, 1, 0, 0]),
    ("DOWN_RIGHT", [0, 1, 0, 1, 0]),
    ("NOTHING", [0] * 5)
)

# Rewards Conf
# +Rewards


# +Penalties
DEATH_PENALTY = -1
RINGS_LOSS_PENALTY = -0.5


def simulate(env, action, show=True):
    if show:
        env.render()

    image, reward, done, _info = env.step([0] * 4 + action + [0] * 3)

    xpos = _info['x']
    ypos = _info['y']

    score = _info["score"]
    rings = _info["rings"]
    lives = _info["lives"]

    screen_x = _info["screen_x"]
    screen_y = _info["screen_y"]
    screen_end_y = _info["screen_x_end"]

    return State(image, xpos, ypos, rings, lives, score, done, reward, screen_x, screen_y, screen_end_y)


def calculate_reward(old_state, new_state):
    if old_state.lives > new_state.lives:
        return DEATH_PENALTY

    if old_state.rings > new_state.rings:
        return RINGS_LOSS_PENALTY

    if old_state.y_position == new_state.y_position and old_state.x_position == new_state.x_position:
        return -0.1

    speed_x = old_state.x_position - new_state.x_position

    reward = -0.01
    if speed_x > 0:
        reward = 0.1
    if old_state.rings < new_state.rings:
        reward = 0.5
    if old_state.score < new_state.score:
        reward = 0.75

    return reward


TRAIN_PLAYS = 100000
TRAIN_CHECKPOINT = 5
PLAY_DURATION = 1200
TARGET_UPDATE = 10

INITIAL_RANDOM_ACTION_RATE = 0.95
DECAY_RATE = 0.001
MIN_RANDOM_ACTION_RATE = 0.05

SKIP_FRAMES = 3

ACTIONS_SIZE = 5
HISTORY_SIZE = 30

# X_SPEED Y_SPEED XPOS YPOS RINGS SCORE END_X X_SCREEN Y_SCREEN
OBSERVATION_SIZE = 9


def train():
    sanic_trainer = GameNetworkTrainer(
        actions_size=ACTIONS_SIZE,
        actions_history_size=HISTORY_SIZE,
        possible_actions=len(ACTIONS),
        observation_size=OBSERVATION_SIZE,
    )

    random_action_rate = INITIAL_RANDOM_ACTION_RATE + INITIAL_RANDOM_ACTION_RATE * DECAY_RATE

    for episode in range(TRAIN_PLAYS):
        ai_bot = sanic_trainer.get_policy_network()

        ai_actions = 0

        if random_action_rate > MIN_RANDOM_ACTION_RATE:
            random_action_rate -= (random_action_rate * DECAY_RATE)
        elif random_action_rate < MIN_RANDOM_ACTION_RATE:
            random_action_rate = MIN_RANDOM_ACTION_RATE

        if episode % TRAIN_CHECKPOINT == 0:
            print("AI GAME ", end=" ")

        env = ENVIRONMENT
        env.reset()

        old_state = simulate(env, ACTIONS[-1][1])

        # observations
        old_actions = [ACTIONS[-1][1] for _ in range(HISTORY_SIZE)]
        x_speed = 0
        y_speed = 0

        #
        play_reward = 0
        # ------------

        for frame in range(PLAY_DURATION):
            observations = [x_speed, y_speed,
                            old_state.x_position, old_state.y_position,
                            old_state.score, old_state.rings,
                            old_state.s_x, old_state.s_y,
                            old_state.end_x]

            world_state = ai_bot.process_state(old_actions, observations, old_state.world)

            if episode % TRAIN_CHECKPOINT == 0 or random.random() > random_action_rate:
                action = ai_bot.get_best_action(world_state)
                ai_actions += 1
            else:
                action = random.randint(0, len(ACTIONS) - 1)

            action_desc, action_command = ACTIONS[action]

            new_state = simulate(env, action_command, show=(episode % TRAIN_CHECKPOINT == 0))

            for _ in range(SKIP_FRAMES):
                new_state = simulate(env, action_command)

            reward = calculate_reward(old_state, new_state)
            play_reward += reward

            x_speed = new_state.x_position - old_state.x_position
            y_speed = new_state.y_position - old_state.y_position

            old_actions.pop(0)
            old_actions.append(action_command)

            old_state = new_state

            observations = [x_speed, y_speed,
                            old_state.x_position, old_state.y_position,
                            old_state.score, old_state.rings,
                            old_state.s_x, old_state.s_y,
                            old_state.end_x]

            new_world_state = ai_bot.process_state(old_actions, observations,old_state.world)

            sanic_trainer.add_replay(world_state, action, reward, new_world_state)

            if episode == 0:
                continue

            sanic_trainer.train()

        if episode % TARGET_UPDATE == 0:
            sanic_trainer.update_target()

        print("Play", episode + 1, "reward", play_reward)
        print("Action rate (", ai_actions / PLAY_DURATION, ")")