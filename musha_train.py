from __future__ import print_function

import json
import random
import time
from collections import namedtuple

import retro

from network_trainer import GameNetworkTrainer

ENVIRONMENT = retro.make("MUSHA-Genesis", "Level1")

State = namedtuple("State", "world lives score done reward")

# ACTION: [0, 0, 0, 0, UP, DOWN, LEFT, RIGHT, JUMP, 0, 0, 0]

ACTIONS = (
    ("UP_SHOOT", [1, 0, 0, 0, 1]),
    ("DOWN_SHOOT", [0, 1, 0, 0, 1]),
    ("RIGHT_SHOOT", [0, 0, 1, 0, 1]),
    ("LEFT_SHOOT", [0, 0, 0, 1, 1]),
    ("NOTHING_SHOOT", [0, 0, 0, 0, 1])
)

ACTIONS_MIRRORED = (
    ("UP_SHOOT", [1, 0, 0, 0, 1]),
    ("DOWN_SHOOT", [0, 1, 0, 0, 1]),
    ("RIGHT_SHOOT", [0, 0, 0, 1, 1]),
    ("LEFT_SHOOT", [0, 0, 1, 0, 1]),
    ("NOTHING_SHOOT", [0, 0, 0, 0, 1])
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

    if old_state.lives < new_state.lives:
        return 1

    score_diff = new_state.score - old_state.score - 30
    score_diff /= 400

    return max(min(score_diff, 1), 0)


class ReplaySaver:

    def __init__(self, backtrack=50, track_bonus=True):
        self.replays = []
        self.replays_bonus = []
        self.backtrack = backtrack
        self.track_bonus = track_bonus
        self.fist_blood = False

    def push_action(self, action, bonus):
        # if self.fist_blood:
        #    return

        if bonus < 0:
            self.fist_blood = True

        self.replays.append(action)
        self.replays_bonus.append(bonus)

    def store_replay(self, replay_path):
        for _ in range(self.backtrack):
            if not self.replays:
                break
            self.replays.pop()
            self.replays_bonus.pop()

        while self.track_bonus and self.replays_bonus and self.replays_bonus[-1] == 0:
            self.replays.pop()
            self.replays_bonus.pop()

        with open(replay_path, "w") as output:
            output.write(json.dumps(self.replays))

    def clean(self):
        self.replays = []
        self.replays_bonus = []
        self.fist_blood = False

    @staticmethod
    def play_replay(replay_path):
        with open(replay_path) as replay_file:
            plays = json.loads(replay_file.read())

        for p in plays:
            yield p


TRAIN_PLAYS = 100000
TRAIN_CHECKPOINT = 5
PLAY_DURATION = 1200
TARGET_UPDATE = 10

INITIAL_RANDOM_ACTION_RATE = 0.95
DECAY_RATE = 0.01
MIN_RANDOM_ACTION_RATE = 0.05

SKIP_FRAMES = 4

ACTIONS_SIZE = 5
HISTORY_SIZE = 30

# X_SPEED Y_SPEED XPOS YPOS RINGS SCORE END_X X_SCREEN Y_SCREEN
OBSERVATION_SIZE = 9

REPLAY_MEMORY = 1150


def train(store_path, load_path=None, start_episode=0, replay_file="replay.rpl"):
    musha_trainer = GameNetworkTrainer(
        possible_actions=len(ACTIONS),
        observation_size=OBSERVATION_SIZE,
        replays_memory=REPLAY_MEMORY,
        save_path=load_path
    )

    random_action_rate = INITIAL_RANDOM_ACTION_RATE + INITIAL_RANDOM_ACTION_RATE * DECAY_RATE

    frames = 0

    for episode in range(start_episode * TRAIN_CHECKPOINT, TRAIN_PLAYS):
        replay_actions = ReplaySaver.play_replay(replay_file)

        musha_pilot = musha_trainer.get_policy_network()

        ai_actions = 0
        replay_recorder = None

        if random_action_rate > MIN_RANDOM_ACTION_RATE:
            random_action_rate -= (random_action_rate * DECAY_RATE)
        elif random_action_rate < MIN_RANDOM_ACTION_RATE:
            random_action_rate = MIN_RANDOM_ACTION_RATE

        ai_game = False
        replay_game = False

        if frames == 0 or episode + 1 % TRAIN_CHECKPOINT == 0:
            print("Replay game")
            replay_game = True
        elif episode % TRAIN_CHECKPOINT == 0:
            print("AI GAME ", end=" ")
            ai_game = True
            replay_recorder = ReplaySaver()

        env = ENVIRONMENT
        env.reset()

        old_state = simulate(env, ACTIONS[-1][1])
        actual_state = simulate(env, ACTIONS[-1][1])

        #
        play_reward = 0
        episode_frame = 0
        # ------------
        nonaction = 0
        mirror_episode = episode + 1 % 2 == 0
        while True:
            frames = min(frames + 1, REPLAY_MEMORY)
            episode_frame += 1

            world_state = musha_pilot.process_state(actual_state.world, old_state.world, flip=mirror_episode)

            if replay_game or ai_game or random.random() > random_action_rate:
                action = next(replay_actions, musha_pilot.get_best_action(world_state).item())
                ai_actions += 1
            else:
                action = next(replay_actions, random.randint(0, len(ACTIONS) - 1))

            action_desc, action_command = ACTIONS[action] if not mirror_episode else ACTIONS_MIRRORED[action]

            old_state = actual_state
            for _ in range(SKIP_FRAMES + 1):
                actual_state = simulate(env, action_command)

            reward = calculate_reward(old_state, actual_state)

            if not reward:
                nonaction += 1
            else:
                nonaction = 0

            if nonaction > 350:
                reward = -5

            if replay_recorder:
                replay_recorder.push_action(action, reward)

            play_reward += reward

            new_world_state = musha_pilot.process_state(actual_state.world, old_state.world, flip=mirror_episode)

            musha_trainer.add_replay(world_state, action, reward, new_world_state)

            if actual_state.done or nonaction > 350:
                break

            if frames < REPLAY_MEMORY:
                continue

            if random.random() > 0.8:
                musha_trainer.train()

        if replay_recorder:
            replay_recorder.store_replay(replay_file)

        if episode % TARGET_UPDATE == 0:
            musha_trainer.update_target()

        print("Play", episode + 1, "reward", play_reward)
        print(f"Action rate: {ai_actions / episode_frame} ({ai_actions} in {episode_frame})")

        musha_trainer.store_policy(store_path + f"/musha_state_{episode // TRAIN_CHECKPOINT}")


def play():
    musha_trainer = GameNetworkTrainer(
        possible_actions=len(ACTIONS),
        observation_size=OBSERVATION_SIZE,
        #       save_path="musha_state"
    )

    musha_pilot = musha_trainer.get_policy_network()
    musha_pilot.set_debug(True)

    env = ENVIRONMENT
    env.reset()

    old_state = simulate(env, ACTIONS[-1][1])
    actual_state = simulate(env, ACTIONS[-1][1])

    print("playing")
    #
    play_reward = 0
    # ------------
    nonaction = 0

    replay_recorder = ReplaySaver()
    mirror_episode = random.choice([True, False])

    for frame in range(PLAY_DURATION):
        world_state = musha_pilot.process_state(actual_state.world, old_state.world,mirror_episode)

        if keyboard.is_pressed("down"):
            action = 1
            time.sleep(3 / 60)
        elif keyboard.is_pressed("left"):
            action = 2
            time.sleep(3 / 60)
        elif keyboard.is_pressed("right"):
            action = 3
            time.sleep(3 / 60)
        elif keyboard.is_pressed("a"):
            action = 4
            time.sleep(3 / 60)
        else:
            action = musha_pilot.get_best_action(world_state).item()

        action_desc, action_command = ACTIONS[action] if not mirror_episode else ACTIONS_MIRRORED[action]

        old_state = actual_state
        for _ in range(SKIP_FRAMES + 1):
            actual_state = simulate(env, action_command, show=True)

        reward = calculate_reward(old_state, actual_state)

        if not reward:
            nonaction += 1
        else:
            nonaction = 0

        if nonaction > 350:
            reward = -5

        if replay_recorder:
            replay_recorder.push_action(action, reward)

        play_reward += reward

        if actual_state.done or nonaction > 350:
            break

    print("Play reward", play_reward)

    replay_recorder.store_replay("musha_replay")
    replay_recorder.clean()

    env = ENVIRONMENT
    env.reset()

    print("Play replay")
    for p in ReplaySaver.play_replay("musha_replay"):
        action_desc, action_command = ACTIONS[p]

        new_state = simulate(env, action_command, show=True)

        for _ in range(SKIP_FRAMES):
            new_state = simulate(env, action_command)

        time.sleep(3 / 60)
        reward = calculate_reward(old_state, new_state)

        if not reward:
            nonaction += 1
        else:
            nonaction = 0

        if nonaction > 350:
            reward = -1

        replay_recorder.push_action(p, reward)

        play_reward += reward

        old_state = new_state

        if old_state.done or nonaction > 350:
            break


if __name__ == '__main__':
    import keyboard

    while True:
        play()
