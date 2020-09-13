#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import unittest

import numpy as np

from envs.azul_env import AzulEnv
from envs.factories import Factories
from envs.wall import Wall


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

    def test_full_column_tile_reward(self):
        wall = Wall(AzulEnv.NUM_COLORS)
        for color_idx in range(AzulEnv.NUM_COLORS):
            wall.reset()
            row_idx = np.arange(AzulEnv.NUM_COLORS)
            column_idx = (row_idx + color_idx) % AzulEnv.NUM_COLORS
            wall.state[row_idx, column_idx] = True
            for i, j in zip(row_idx, column_idx):
                self.assertEqual(wall.compute_build_reward(i, j), 11)


if __name__ == '__main__':
    unittest.main()
