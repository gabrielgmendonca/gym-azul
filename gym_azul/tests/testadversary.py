#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import unittest
from unittest.mock import Mock

from gym_azul.envs.azul_env import AzulEnv
from gym_azul.envs.factories import Factories
from gym_azul.envs.wall import Wall

from gym_azul.envs.adversary import *


class TestAdversary(unittest.TestCase):
    def test_random_first_play(self):
        np_random = Mock()
        np_random.randint.return_value = 0
        factories = Mock()
        factories.num_colors = 5
        factories.num_factories = 5
        factories.pick_tiles.return_value = 2, False, True
        adversary = RandomAdversary(factories, np_random)
        adversary.wall = Mock()
        adversary.wall.add_tiles.return_value = 10
        adversary.wall.done.return_value = False

        round_end = adversary.play()
        self.assertFalse(round_end)
        self.assertFalse(adversary.done())
        self.assertEqual(adversary.score, 10)
        factories.pick_tiles.assert_called_once_with(0, 0)
        adversary.wall.add_tiles.assert_called_once_with(0, 0, 2, True)

    def test_done(self):
        factories = Mock()
        factories.num_colors = 5
        adversary = RandomAdversary(factories, Mock())
        adversary.wall = Mock()
        adversary.wall.done.return_value = True
        self.assertTrue(adversary.done())

    def test_reset(self):
        factories = Mock()
        factories.num_colors = 5
        adversary = RandomAdversary(factories, Mock())
        adversary.wall = Mock()
        adversary.reset()
        adversary.wall.reset.assert_called_once()

    def test_end_round(self):
        factories = Mock()
        factories.num_colors = 5
        adversary = RandomAdversary(factories, Mock())
        adversary.wall = Mock()
        adversary.wall.floor_state = 7
        adversary.end_round()
        self.assertEqual(adversary.wall.floor_state, 0)


if __name__ == '__main__':
    unittest.main()
