#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from envs.factories import Factories
from envs.wall import Wall


class AzulEnv(gym.Env):
    metadata = {'render.modes': ['console', 'human']}
    NUM_COLORS = 5
    NUM_FACTORIES = 5
    FACTORY_SIZE = 4
    EMPTY_PICK_REWARD = -1

    def __init__(self):
        super(AzulEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([self.NUM_FACTORIES + 1,
                                                  self.NUM_COLORS,
                                                  self.NUM_COLORS])
        self.factories = Factories(self.NUM_COLORS, self.FACTORY_SIZE,
                                   self.NUM_FACTORIES)
        self.wall = Wall(self.NUM_COLORS)
        self.adversary_wall = Wall(self.NUM_COLORS)
        self.observation_space = spaces.MultiDiscrete(
            self.factories.state_space + self.wall.state_space)
        self.seed()
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        reward = 0
        info = {}
        num_tiles, round_end = self.factories.pick_tiles(action[0], action[1])
        if num_tiles > 0:
            reward += self.wall.add_tiles(action[1], action[2], num_tiles)
            if round_end:
                self.end_round()
            round_end = self.adversary_play()
            if round_end:
                self.end_round()
        else:
            reward += EMPTY_PICK_REWARD
            info = {'info': 'empty pick'}

        observation = np.concatenate((self.factories.get_observation(),
                                      self.wall.get_observation()))
        done = self.wall.done() or self.adversary_wall.done()
        return observation, reward, done, info

    def reset(self):
        self.factories.reset()
        self.wall.reset()
        self.adversary_wall.reset()
        observation = np.concatenate((self.factories.get_observation(),
                                      self.wall.get_observation()))
        return observation

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
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

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def adversary_play(self):
        # TODO: Use trained model as adversary
        num_tiles = 0
        while num_tiles == 0:
            factory_idx = np.random.randint(self.NUM_FACTORIES + 1)
            color_idx = np.random.randint(self.NUM_COLORS)
            num_tiles, round_end = self.factories.pick_tiles(factory_idx,
                                                             color_idx)
        self.adversary_wall.add_tiles(factory_idx, color_idx, num_tiles)
        return round_end

    def end_round(self):
        self.factories.reset()
        self.wall.floor_state = 0
        self.adversary_wall.floor_state = 0
