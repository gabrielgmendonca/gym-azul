#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Gabriel Mendonça

import matplotlib.pyplot as plt
import numpy as np


class Board:
    COLOR_MAP = np.array([[60, 140, 180, 16],
                          [240, 180, 50, 32],
                          [240, 40, 65, 16],
                          [40, 50, 55, 16],
                          [200, 235, 230, 32]])

    def __init__(self, height, width, num_colors):
        self.height = height
        self.width = width
        self.num_colors = num_colors
        self.figure = plt.figure(figsize=(5, 5))
        cid = self.figure.canvas.mpl_connect('button_press_event', self.onclick)

    def render(self, factories, wall):
        self.img = np.ones((self.height, self.width, 4)).astype(int) * 255
        self._render_wall(wall)
        self._render_pattern_line(wall)
        return self.img

    def show(self):
        plt.clf()
        plt.imshow(self.img)
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.pause(1)

    def close(self):
        plt.close()

    def reset(self):
        self.img = np.ones((self.width, self.height, 4)).astype(int) * 255

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))

    def _render_wall(self, wall):
        x_start = round(self.width / 2)
        y_start = round(self.height / 2)
        canvas = self.img[x_start:, y_start:]

        for color_idx in range(self.num_colors):
            for row_idx in range(self.num_colors):
                size = round(self.width / 2 / self.num_colors)
                pos = size * row_idx
                next_pos = pos + size

                is_complete = wall.is_complete(color_idx, row_idx)
                color = self.COLOR_MAP[color_idx].copy()
                if is_complete:
                    color[3] = 255
                self._fill(canvas, pos, pos + size * color_idx,
                           next_pos, next_pos + size * color_idx, color)

        for i in range(0, self.num_colors):
            pos = round(self.width / 2 / self.num_colors * i)
            self._draw_vline(canvas, pos)
            self._draw_hline(canvas, pos)

    def _render_pattern_line(self, wall):
        x_start = round(self.height / 2)
        y_end = round(self.width / 2)
        canvas = self.img[x_start:, :y_end]

        tile_size = round(self.width / 2 / self.num_colors)
        for i in range(self.num_colors):
            xs = tile_size * i
            xe = xs + tile_size

            color_idx = wall.pattern_line_state[1, i]
            color = self.COLOR_MAP[color_idx].copy()
            color[3] = 255

            num_tiles = wall.pattern_line_state[0, i]
            for j in range(num_tiles):
                ys = -tile_size * (j + 1)
                ye = ys + tile_size - 1
                self._fill(canvas, xs, ys, xe, ye, color)

        for i in range(self.num_colors):
            pos = tile_size * i
            offset = pos + tile_size
            self._draw_vline(canvas[-offset:], pos)
            self._draw_hline(canvas[:, -offset:], pos)

    def _draw_vline(self, canvas, pos):
        canvas[:, pos] = [0, 0, 0, 255]

    def _draw_hline(self, canvas, pos):
        canvas[pos] = [0, 0, 0, 255]

    def _fill(self, canvas, xs, ys, xe, ye, color):
        if ys >= canvas.shape[1]:
            ys -= canvas.shape[1]
        if ye > canvas.shape[1]:
            ye -= canvas.shape[1]

        canvas[xs:xe, ys:ye] = color
