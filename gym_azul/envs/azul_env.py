#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from factories import Factories
from wall import wall


class AzulEnv(gym.Env):
    metadata = {'render.modes': ['console', 'human']}
    NUM_COLORS = 5
    NUM_FACTORIES = 5
    FACTORY_SIZE = 4

    def __init__(self):
        super(AzulEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([self.NUM_FACTORIES + 1,
                                                  self.NUM_COLORS,
                                                  self.NUM_COLORS])
        self.factories = Factories(self.NUM_COLORS, self.FACTORY_SIZE,
                                   self.NUM_FACTORIES)
        self.wall = Wall(self.NUM_COLORS)
        self.observation_space = spaces.MultiDiscrete(
            self.factories.state_space + self.wall.state_space)
        self.seed()
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
#         print('F {} | C {} | R {}'.format(*action))

        reward = 0
        info = {}
        num_tiles = self.factories.pick_tiles(action[0], action[1])
        if num_tiles > 0:
            reward += self.wall.add_tiles(action[1], action[2], num_tiles)
            self.adversary_play()
        else:
            reward -= 0.1
            info = {'info': 'empty pick'}

        observation = np.concatenate((self.factories.get_observation(),
                                      self.wall.get_observation()))
        done = self.wall.done()
        return observation, reward, done, info

    def reset(self):
        self.factories.reset()
        self.wall.reset()
        observation = np.concatenate((self.factories.get_observation(),
                                      self.wall.get_observation()))
        return observation

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print()
        print('-' * 15)
        print('Factories:')
        print(self.factories.get_observation()[:-1].reshape((3, 5)))
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
        num_tiles = 0
        while num_tiles == 0:
            factory_idx = np.random.randint(self.NUM_FACTORIES + 1)
            color_idx = np.random.randint(self.NUM_COLORS)
            num_tiles = self.factories.pick_tiles(factory_idx, color_idx)
#         print('ADV: F {} | C {}'.format(factory_idx, color_idx))
