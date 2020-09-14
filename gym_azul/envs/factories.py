#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import numpy as np


class Factories:
    TABLE_SIZE = 20

    def __init__(self, num_colors, size, num_factories):
        self.num_colors = num_colors
        self.size = size
        self.num_factories = num_factories
        self.state = np.zeros([self.num_factories + 1, self.num_colors],
                               dtype=np.uint8)
        self.state_space = [self.TABLE_SIZE + 1] * self.num_colors + \
            [self.size + 1] * self.num_colors * self.num_factories + [2]
        self.reset()

    def reset(self):
        self.state[0] = 0
        self.first_player_table = True
        for i in range(1, self.num_factories + 1):
            tiles = np.random.randint(self.num_colors, size=self.size)
            self.state[i] = np.bincount(tiles, minlength=self.num_colors)
            assert(np.sum(self.state[i]) == self.size)

    def pick_tiles(self, factory_idx, color_idx):
        # TODO: first_player token
        round_end = False
        num_tiles = self.state[factory_idx, color_idx]
        if num_tiles > 0:
            self.state[factory_idx, color_idx] = 0
            if factory_idx != 0:
                self.state[0] += self.state[factory_idx]  # move to table
                self.state[factory_idx] = 0

        if np.sum(self.state) == 0:
            round_end = True
        
        return num_tiles, round_end

    def get_observation(self):
        observation = np.ndarray.flatten(self.state)
        observation = np.concatenate((observation,
            [int(self.first_player_table)]))
        return observation
