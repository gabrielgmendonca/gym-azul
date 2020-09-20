#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import matplotlib.pyplot as plt
import numpy as np


class Board:
    COLOR_MAP = np.array([[60, 140, 180, 32],
                          [240, 180, 50, 32],
                          [240, 40, 65, 32],
                          [40, 50, 55, 32],
                          [200, 235, 230, 32]])

    def __init__(self, height, width, num_colors):
        self.height = height
        self.width = width
        self.num_colors = num_colors

    def render(self, factories, wall):
        self.img = np.ones((self.height, self.width, 4)).astype(int) * 255
        self._render_wall(wall)
        return self.img

    def show(self):
        # plt.figure(figsize=(10, 10))
        plt.clf()
        plt.imshow(self.img)
        plt.pause(1)

    def close(self):
        plt.close()

    def reset(self):
        self.img = np.ones((self.height, self.width, 4)).astype(int) * 255
        plt.close()

    def _draw_vline(self, pos):
        self.img[:, pos] = [0, 0, 0, 255]

    def _draw_hline(self, pos):
        self.img[pos] = [0, 0, 0, 255]

    def _fill(self, xs, ys, xe, ye, color):
        if ys >= self.img.shape[1]:
            ys -= self.img.shape[1]
        if ye > self.img.shape[1]:
            ye -= self.img.shape[1]
        
        self.img[xs:xe, ys:ye] = color

    def _render_wall(self, wall):
        for color_idx in range(self.num_colors):
            for row_idx in range(self.num_colors):
                size = round(self.width / self.num_colors)
                pos = size * row_idx
                next_pos = pos + size

                is_complete = wall.is_complete(color_idx, row_idx)
                color = self.COLOR_MAP[color_idx].copy()
                if is_complete:
                    color[3] = 255
                self._fill(pos, pos + size * color_idx,
                           next_pos, next_pos + size * color_idx, color)

        for i in range(1, self.num_colors):
            pos = round(self.width / self.num_colors * i)
            self._draw_vline(pos)
            self._draw_hline(pos)
