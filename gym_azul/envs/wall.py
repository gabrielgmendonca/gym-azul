#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

class Wall:
    FLOOR_LIMIT = 7

    def __init__(self, num_colors):
        self.num_colors = num_colors
        self.reset()
        self.state_space = ([2] * self.num_colors * self.num_colors +  # wall
                            list(range(2, self.num_colors + 2)) +      # pattern lines num
                            [self.num_colors] * self.num_colors +      # pattern lines color
                            [self.FLOOR_LIMIT + 1])                    # floor

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
        # TODO: Implement scoring
        column_idx = (row_idx + color_idx) % self.num_colors
        self.state[row_idx, column_idx] = 1
        self.pattern_line_state[0, row_idx] = 0
        return 5

    def break_tiles(self, num_tiles):
        # TODO: Add correct penalty for breaking
#         previous = self.floor_state
#         self.floor_state = min(self.floor_state + num_tiles, self.FLOOR_LIMIT)
#         num_broken = self.floor_state - previous
#         reward = -1 * num_broken
        reward = -1 * num_tiles
        return reward

    def done(self):
        return bool(self.state.all(axis=1).any())

    def get_observation(self):
        observation = np.concatenate((np.ndarray.flatten(self.state),
            np.ndarray.flatten(self.pattern_line_state), [self.floor_state]))
        return observation
