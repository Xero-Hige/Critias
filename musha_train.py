from __future__ import print_function

from collections import namedtuple

import retro
import numpy as np
import cv2
import pickle
import random

from network_trainer import GameNetworkTrainer

# random.seed(99)

ENVIRONMENT = retro.make("MUSHA-Genesis", "Level1")

State = namedtuple("State", "world lives score done reward")

# ACTION: [0, 0, 0, 0, UP, DOWN, LEFT, RIGHT, JUMP, 0, 0, 0]

ACTIONS = (
    ("UP", [1, 0, 0, 0, 0]),
    ("DOWN", [0, 1, 0, 0, 0]),
    ("LEFT", [0, 0, 1, 0, 0]),
    ("RIGHT", [0, 0, 0, 1, 0]),
    ("SHOOT", [0, 0, 0, 0, 1]),
    ("UP_SHOOT", [1, 0, 0, 0, 1]),
    ("DOWN_SHOOT", [0, 1, 0, 0, 1]),
    ("RIGHT_SHOOT", [0, 0, 1, 0, 1]),
    ("LEFT_SHOOT", [0, 0, 0, 1, 1]),
    ("NOTHING", [0] * 5)
)


def simulate(env, action, show=False):
    if show:
        env.render()

    image, reward, done, _info = env.step([0] * 4 + action + [0] * 3)

    score = _info["score"]
    lives = _info["lives"]

    return State(image, lives, score, done, reward)


def calculate_reward(old_state, new_state):
    if old_state.lives > new_state.lives:
        return -1

    return 1 if old_state.score < old_state.score else 0


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


def train(store_path):
    musha_trainer = GameNetworkTrainer(
        possible_actions=len(ACTIONS),
        observation_size=OBSERVATION_SIZE,
        replays_memory=550
    )

    random_action_rate = INITIAL_RANDOM_ACTION_RATE + INITIAL_RANDOM_ACTION_RATE * DECAY_RATE

    frames = 0

    for episode in range(TRAIN_PLAYS):
        musha_pilot = musha_trainer.get_policy_network()

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

        #
        play_reward = 0
        # ------------
        nonaction = 0

        while True:

            frames = min(frames + 1, 550)

            world_state = musha_pilot.process_state(old_state.world)

            if episode % TRAIN_CHECKPOINT == 0 or random.random() > random_action_rate:
                action = musha_pilot.get_best_action(world_state).item()
                ai_actions += 1
            else:
                action = random.randint(0, len(ACTIONS) - 1)

            action_desc, action_command = ACTIONS[action]

            new_state = simulate(env, action_command, show=False)

            for _ in range(SKIP_FRAMES):
                new_state = simulate(env, action_command)

            reward = calculate_reward(old_state, new_state)

            if not reward:
                nonaction += 1

            play_reward += reward

            new_world_state = musha_pilot.process_state(old_state.world)

            musha_trainer.add_replay(world_state, action, reward, new_world_state)

            old_state = new_state

            if old_state.done or nonaction > 600:
                break

            if frames < 550:
                continue

            musha_trainer.train()

        if episode % TARGET_UPDATE == 0:
            musha_trainer.update_target()

        print("Play", episode + 1, "reward", play_reward)
        print("Action rate (", ai_actions / PLAY_DURATION, ")")

        musha_trainer.store_policy(store_path + f"/musha_state_{episode // TRAIN_CHECKPOINT}")


def play():
    musha_trainer = GameNetworkTrainer(
        possible_actions=len(ACTIONS),
        observation_size=OBSERVATION_SIZE
    )

    musha_pilot = musha_trainer.get_policy_network()

    env = ENVIRONMENT
    env.reset()

    old_state = simulate(env, ACTIONS[-1][1])

    old_actions = [ACTIONS[-1][1] for _ in range(HISTORY_SIZE)]
    print("playing")
    #
    play_reward = 0
    # ------------
    nonaction = 0

    for frame in range(PLAY_DURATION):
        world_state = musha_pilot.process_state(old_state.world)

        action = musha_pilot.get_best_action(world_state)

        action_desc, action_command = ACTIONS[action]

        new_state = simulate(env, action_command, show=True)

        for _ in range(SKIP_FRAMES):
            new_state = simulate(env, action_command)

        reward = calculate_reward(old_state, new_state)

        if not reward:
            nonaction += 1

        play_reward += reward

        old_actions.pop(0)
        old_actions.append(action_command)

        old_state = new_state

        if old_state.done or nonaction > 600:
            break

    print("Play reward", play_reward)
