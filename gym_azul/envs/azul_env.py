#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from .adversary import RandomAdversary, PPO2Adversary
from .board import Board
from .factories import Factories
from .wall import Wall


class AzulEnv(gym.Env):
    metadata = {'render.modes': ['console', 'human']}
    NUM_COLORS = 5
    NUM_FACTORIES = 5
    FACTORY_SIZE = 4
    EMPTY_PICK_REWARD = -10
    MAX_ACTIONS = NUM_COLORS * sum(range(1, NUM_COLORS + 1))

    def __init__(self, adv_model_path=None, reward_type='score'):
        super().__init__()
        self.seed()
        self.action_space = spaces.MultiDiscrete([self.NUM_FACTORIES + 1,
                                                  self.NUM_COLORS,
                                                  self.NUM_COLORS])
        self.factories = Factories(self.NUM_COLORS, self.FACTORY_SIZE,
                                   self.NUM_FACTORIES, self.np_random)
        self.wall = Wall(self.NUM_COLORS)
        self.observation_space = spaces.MultiDiscrete(
            self.factories.state_space + self.wall.state_space)
        self.board = None

        if adv_model_path:
            self.adversary = PPO2Adversary(self.factories, self.np_random,
                                           adv_model_path)
        else:
            self.adversary = RandomAdversary(self.factories, self.np_random)
        self.reward_type = reward_type
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        self.num_actions += 1
        reward = 0
        info = {}

        num_tiles, round_end, first_player_token = \
            self.factories.pick_tiles(action[0], action[1])

        if num_tiles > 0:
            action_score = self.wall.add_tiles(
                action[1], action[2], num_tiles, first_player_token)
            self.score += action_score

            if self.reward_type == 'score':
                reward = action_score

            if round_end:
                self.end_round()
            round_end = self.adversary.play()
            if round_end:
                self.end_round()

        else:
            info['info'] = 'empty pick'
            if self.reward_type == 'score':
                reward = self.EMPTY_PICK_REWARD

        observation = np.concatenate((self.factories.get_observation(),
                                      self.wall.get_observation()))
        done = (self.wall.done() or self.adversary.done() or
                self.num_actions >= self.MAX_ACTIONS)

        if done and self.reward_type == 'win':
            if (self.score <= self.adversary.score or
                self.num_actions >= self.MAX_ACTIONS):
                reward = -1
            else:
                reward = 1
        return observation, reward, done, info

    def reset(self):
        self.num_actions = 0
        self.score = 0
        self.factories.reset()
        self.wall.reset()
        self.adversary.reset()
        if self.board:
            self.board.reset()
        observation = np.concatenate((self.factories.get_observation(),
                                      self.wall.get_observation()))
        return observation

    def render(self, mode='console', close=False):
        if close:
            if self.board:
                self.board.close()
            return

        if mode == 'console':
            print()
            print('-' * 15)
            print('Factories:')
            print(self.factories.get_observation()[:-1].reshape(
                (self.NUM_FACTORIES + 1, self.NUM_COLORS)))
            print()
            print('Wall:')
            for i in range(self.NUM_COLORS):
                print(str(self.wall.pattern_line_state[0, i]),
                      str(self.wall.state[i].astype(int)))
            print()
            return

        if not self.board:
            self.board = Board(1024, 1024, self.NUM_COLORS)

        img = self.board.render(self.factories, self.wall)
        if mode == 'human':
            self.board.show()

        return img

    def close(self):
        if self.board:
            self.board.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def end_round(self):
        self.factories.reset()
        self.wall.floor_state = 0
        self.adversary.end_round()
