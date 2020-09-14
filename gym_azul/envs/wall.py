#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import numpy as np


class Wall:
    FLOOR_PENALTY = np.array([-1] * 2 + [-2] * 3 + [-3] * 2)

    def __init__(self, num_colors):
        self.num_colors = num_colors
        self.reset()
        self.state_space = ([2] * self.num_colors * self.num_colors +  # wall
                            list(range(2, self.num_colors + 2)) +      # pattern lines num
                            [self.num_colors] * self.num_colors +      # pattern lines color
                            [len(self.FLOOR_PENALTY) + 1])             # floor

    def reset(self):
        self.state = np.zeros([self.num_colors, self.num_colors], dtype=bool)
        self.pattern_line_state = np.zeros([2, self.num_colors], dtype=np.uint8)
        self.floor_state = 0
        
    def add_tiles(self, color_idx, row_idx, num_tiles):
        assert(num_tiles > 0)
        reward = 0
        if self.is_complete(color_idx, row_idx):  # already on the wall
            reward += self.break_tiles(num_tiles)
        elif (self.pattern_line_state[0, row_idx] > 0 and
              self.pattern_line_state[1, row_idx] != color_idx):  # pattern line in use
            reward += self.break_tiles(num_tiles)
        else:  # empty or correct pattern line
            self.pattern_line_state[1, row_idx] = color_idx
            available_space = row_idx + 1 - self.pattern_line_state[0, row_idx]
            if num_tiles >= available_space:  # enough to build
                reward += self.build_tile(color_idx, row_idx)
                excess = num_tiles - available_space
                if excess > 0:
                    reward += self.break_tiles(excess)
            else:
                self.pattern_line_state[0, row_idx] += num_tiles
        return reward

    def is_complete(self, color_idx, row_idx):
        column_idx = (row_idx + color_idx) % self.num_colors
        return self.state[row_idx, column_idx]

    def build_tile(self, color_idx, row_idx):
        column_idx = (row_idx + color_idx) % self.num_colors
        self.state[row_idx, column_idx] = 1
        self.pattern_line_state[0, row_idx] = 0
        return self.compute_build_reward(row_idx, column_idx)

    def compute_build_reward(self, row_idx, column_idx):
        reward = 0
        row_adj = self._count_adjacent(self.state[row_idx], column_idx)
        col_adj = self._count_adjacent(self.state[:, column_idx], row_idx)

        if col_adj == 1:
            reward = row_adj
        elif row_adj == 1:
            reward = col_adj
        else:
            reward = row_adj + col_adj

        if row_adj == self.num_colors:  # Full row
            reward += 2
        if col_adj == self.num_colors:  # Full column
            reward += 7

        color_idx = (column_idx - row_idx) % self.num_colors
        num_same_color = np.sum(
            [self.is_complete(color_idx, i) for i in range(self.num_colors)])
        if num_same_color == 5:
            reward += 10
        return reward

    def break_tiles(self, num_tiles):
        previous = self.floor_state
        self.floor_state = min(self.floor_state + num_tiles,
                               len(self.FLOOR_PENALTY))
        return self.FLOOR_PENALTY[previous:self.floor_state].sum()

    def done(self):
        return bool(self.state.all(axis=1).any())

    def get_observation(self):
        observation = np.concatenate((np.ndarray.flatten(self.state),
            np.ndarray.flatten(self.pattern_line_state), [self.floor_state]))
        return observation

    def _count_adjacent(self, line, idx):
        change_points = np.where(np.diff(line) != 0)[0] + 1
        tile_sequences = np.split(np.arange(len(line)), change_points)
        for seq in tile_sequences:
            if idx in seq:
                return len(seq)
        raise ValueError
