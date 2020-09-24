#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

from abc import ABC, abstractmethod

from gym_azul.envs.wall import Wall


class Adversary(ABC):
    def __init__(self, factories, np_random):
        self.factories = factories
        self.np_random = np_random
        self.wall = Wall(factories.num_colors)
        self.score = 0

    @abstractmethod
    def play(self):
        pass

    def done(self):
        return self.wall.done()

    def reset(self):
        self.wall.reset()

    def end_round(self):
        self.wall.floor_state = 0

class RandomAdversary(Adversary):
    def play(self):
        num_tiles = 0
        while num_tiles == 0:
            factory_idx = \
                self.np_random.randint(self.factories.num_factories + 1)
            color_idx = self.np_random.randint(self.factories.num_colors)
            num_tiles, round_end, first_player_token = \
                self.factories.pick_tiles(factory_idx, color_idx)

        row_idx = self.np_random.randint(self.factories.num_colors)
        self.score += self.wall.add_tiles(
            color_idx, row_idx, num_tiles, first_player_token)
        return round_end
