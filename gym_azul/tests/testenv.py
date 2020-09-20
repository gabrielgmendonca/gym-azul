#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import unittest

import numpy as np

from gym_azul.envs.azul_env import AzulEnv
from gym_azul.envs.factories import Factories
from gym_azul.envs.wall import Wall


class TestWall(unittest.TestCase):
    def test_single_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for i in range(AzulEnv.NUM_COLORS):
            for j in range(AzulEnv.NUM_COLORS):
                wall.reset()
                wall.state[i, j] = True
                self.assertEqual(wall.compute_build_reward(i, j), 1)

    def test_adjecent_row_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for k in range(2, AzulEnv.NUM_COLORS):
            for i in range(AzulEnv.NUM_COLORS):
                for j in range(AzulEnv.NUM_COLORS - k + 1):
                    wall.reset()
                    wall.state[i, j:j + k] = True
                    for kk in range(k):
                        self.assertEqual(
                            wall.compute_build_reward(i, j + kk), k)

    def test_adjecent_column_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for k in range(2, AzulEnv.NUM_COLORS):
            for i in range(AzulEnv.NUM_COLORS):
                for j in range(AzulEnv.NUM_COLORS - k + 1):
                    wall.reset()
                    wall.state[j:j + k, i] = True
                    for kk in range(k):
                        self.assertEqual(
                            wall.compute_build_reward(j + kk, i), k)

    def test_adjecent_row_and_column_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for k in range(2, AzulEnv.NUM_COLORS):
            for i in range(AzulEnv.NUM_COLORS - k + 1):
                for j in range(AzulEnv.NUM_COLORS - k + 1):
                    wall.reset()
                    wall.state[i:i + k, j:j + k] = True
                    for ki in range(k):
                        for kj in range(k):
                            self.assertEqual(
                                wall.compute_build_reward(i + ki, j + kj),
                                k * 2)

    def test_full_row_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for i in range(AzulEnv.NUM_COLORS):
            wall.reset()
            wall.state[i, :] = True
            for j in range(AzulEnv.NUM_COLORS):
                self.assertEqual(wall.compute_build_reward(i, j), 7)

    def test_full_column_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for i in range(AzulEnv.NUM_COLORS):
            wall.reset()
            wall.state[:, i] = True
            for j in range(AzulEnv.NUM_COLORS):
                self.assertEqual(wall.compute_build_reward(j, i), 12)

    def test_full_color_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for color_idx in range(AzulEnv.NUM_COLORS):
            wall.reset()
            row_idx = np.arange(AzulEnv.NUM_COLORS)
            column_idx = (row_idx + color_idx) % AzulEnv.NUM_COLORS
            wall.state[row_idx, column_idx] = True
            for i, j in zip(row_idx, column_idx):
                self.assertEqual(wall.compute_build_reward(i, j), 11)

    def test_break_one_tile(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        self.assertEqual(wall.break_tiles(1), -1)
        self.assertEqual(wall.break_tiles(1), -1)
        self.assertEqual(wall.break_tiles(1), -2)
        self.assertEqual(wall.break_tiles(1), -2)
        self.assertEqual(wall.break_tiles(1), -2)
        self.assertEqual(wall.break_tiles(1), -3)
        self.assertEqual(wall.break_tiles(1), -3)

    def test_break_two_tiles(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        self.assertEqual(wall.break_tiles(2), -2)
        self.assertEqual(wall.break_tiles(2), -4)
        self.assertEqual(wall.break_tiles(2), -5)
        self.assertEqual(wall.break_tiles(2), -3)

    def test_break_three_tiles(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        self.assertEqual(wall.break_tiles(3), -4)
        self.assertEqual(wall.break_tiles(3), -7)
        self.assertEqual(wall.break_tiles(3), -3)

    def test_break_four_tiles(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        self.assertEqual(wall.break_tiles(4), -6)
        self.assertEqual(wall.break_tiles(4), -8)

    def test_break_five_tiles(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        self.assertEqual(wall.break_tiles(5), -8)
        self.assertEqual(wall.break_tiles(5), -6)

    def test_add_tiles_first_player_token(self):
        wall = Wall(AzulEnv.NUM_COLORS)

        first_player_token = False
        reward = wall.add_tiles(0, 0, 1, first_player_token)
        self.assertEqual(reward, 1)

        first_player_token = True
        reward = wall.add_tiles(2, 0, 1, first_player_token)
        self.assertEqual(reward, 0)

        first_player_token = True
        reward = wall.add_tiles(2, 0, 1, first_player_token)
        self.assertEqual(reward, -3)


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
