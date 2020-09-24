#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import unittest

import numpy as np

from gym_azul.envs.azul_env import AzulEnv
from gym_azul.envs.factories import Factories


class TestFactories(unittest.TestCase):
    def _pick_first(self, factories):
        factory_idx, color_idx = np.unravel_index(
            (factories.state > 0).argmax(), factories.state.shape)
        return factories.pick_tiles(factory_idx, color_idx)

    def test_round_end(self):
        factories = Factories(AzulEnv.NUM_COLORS, AzulEnv.FACTORY_SIZE,
            AzulEnv.NUM_FACTORIES, np.random.RandomState())
        while factories.state.sum() > 0:
            num_tiles, round_end, _ = self._pick_first(factories)
            self.assertGreater(num_tiles, 0)
        self.assertTrue(round_end)

    def test_pick_tiles_first_player(self):
        factories = Factories(AzulEnv.NUM_COLORS, AzulEnv.FACTORY_SIZE,
            AzulEnv.NUM_FACTORIES, np.random.RandomState())

        num_tiles, round_end, first_player_token = self._pick_first(factories)
        self.assertGreater(num_tiles, 0)
        self.assertFalse(round_end)
        self.assertFalse(first_player_token)

        num_tiles, round_end, first_player_token = self._pick_first(factories)
        self.assertGreater(num_tiles, 0)
        self.assertFalse(round_end)
        self.assertTrue(first_player_token)

        num_tiles, round_end, first_player_token = self._pick_first(factories)
        self.assertGreater(num_tiles, 0)
        self.assertFalse(round_end)
        self.assertFalse(first_player_token)


if __name__ == '__main__':
    unittest.main()
