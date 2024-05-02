from cv2 import imread, imwrite

import functools
import random
from copy import copy

from gymnasium.spaces.space import Space
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv

class Map(ParallelEnv):
    metadata = {
        "name" : "custom_environment"
    }

    def __init__(self, file_name, max_timesteps = 2000):
        self.map = imread(file_name)
        self.escape_x = None
        self.escape_y = None
        self.guard_x = None
        self.guard_y = None
        self.prisoner_x = None
        self.prisoner_y = None
        self.timestep = None
        self.max_timesteps = max_timesteps if max_timesteps is not None else 2000
        self.x_lim = self.map.shape[1]
        self.y_lim = self.map.shape[0]
        self.possible_agents = ["guard", "prisoner"]

    def reset(self, seed = None, options = None):
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = self.map.shape[0] - 1
        self.guard_y = self.map.shape[1] - 1

        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y][x][0] == 255 and self.map[y][x][1] == 0 and self.map[y][x][2] == 0:
                    self.escape_x = x
                    self.escape_y = y
        
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y][x][0] == 255 and self.map[y][x][1] == 0 and self.map[y][x][2] == 0 and (y != self.escape_y or x != self.escape_x):
                    self.map[y][x][0] = 255
                    self.map[y][x][1] = 255
                    self.map[y][x][2] = 255
        
        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": {"observation": observation, "action_mask": [0, 1, 1, 0]},
            "guard": {"observation": observation, "action_mask": [1, 0, 0, 1]},
        }

        infos = {
            a : {} for a in self.agents
        }
        self.render()
        
        return observations, infos
    
    def step(self, actions):
        prisoner_action = actions["prisoner"]
        guard_action = actions["guard"]
        prisoner_prev_position = (self.prisoner_x, self.prisoner_y)
        guard_prev_position = (self.guard_x, self.guard_y)
        
        if prisoner_action == 0 and self.prisoner_x > 0:
            if (
                self.map[self.prisoner_y][self.prisoner_x - 1][0] != 0 or
                self.map[self.prisoner_y][self.prisoner_x - 1][1] != 0 or
                self.map[self.prisoner_y][self.prisoner_x - 1][2] != 0
            ):
                self.prisoner_x -= 1
        if prisoner_action == 1 and self.prisoner_x < self.x_lim - 1:
            if (
                self.map[self.prisoner_y][self.prisoner_x + 1][0] != 0 or
                self.map[self.prisoner_y][self.prisoner_x + 1][1] != 0 or
                self.map[self.prisoner_y][self.prisoner_x + 1][2] != 0
            ):
                self.prisoner_x += 1
        if prisoner_action == 2 and self.prisoner_y > 0:
            if (
                self.map[self.prisoner_y - 1][self.prisoner_x][0] != 0 or
                self.map[self.prisoner_y - 1][self.prisoner_x][1] != 0 or
                self.map[self.prisoner_y - 1][self.prisoner_x][2] != 0
            ):
                self.prisoner_y -= 1
        if prisoner_action == 3 and self.prisoner_y < self.y_lim - 1:
            if (
                self.map[self.prisoner_y + 1][self.prisoner_x][0] != 0 or
                self.map[self.prisoner_y + 1][self.prisoner_x][1] != 0 or
                self.map[self.prisoner_y + 1][self.prisoner_x][2] != 0
            ):
                self.prisoner_y += 1
            
        if guard_action == 0 and self.guard_x > 0:
            if (
                self.map[self.guard_y][self.guard_x - 1][0] != 0 or
                self.map[self.guard_y][self.guard_x - 1][1] != 0 or
                self.map[self.guard_y][self.guard_x - 1][2] != 0
            ):
                self.guard_x -= 1
        if guard_action == 1 and self.guard_x < self.x_lim - 1:
            if (
                self.map[self.guard_y][self.guard_x + 1][0] != 0 or
                self.map[self.guard_y][self.guard_x + 1][1] != 0 or
                self.map[self.guard_y][self.guard_x + 1][2] != 0
            ):
                self.guard_x += 1
        if guard_action == 2 and self.guard_y > 0:
            if (
                self.map[self.guard_y - 1][self.guard_x][0] != 0 or
                self.map[self.guard_y - 1][self.guard_x][1] != 0 or
                self.map[self.guard_y - 1][self.guard_x][2] != 0
            ):
                self.guard_y -= 1
        if guard_action == 3 and self.guard_y < self.y_lim - 1:
            if (
                self.map[self.guard_y + 1][self.guard_x][0] != 0 or
                self.map[self.guard_y + 1][self.guard_x][1] != 0 or
                self.map[self.guard_y + 1][self.guard_x][2] != 0
            ):
                self.guard_y += 1
            
        prisoner_action_mask = np.ones(4, dtype = np.int8)
        if self.prisoner_x == 0:
            prisoner_action_mask[0] = 0
        elif self.prisoner_x == 6:
            prisoner_action_mask[1] = 0
        if self.prisoner_y == 0:
            prisoner_action_mask[2] = 0
        elif self.prisoner_y == 6:
            prisoner_action_mask[3] = 0

        guard_action_mask = np.ones(4, dtype = np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0
        elif self.guard_x == 6:
            guard_action_mask[1] = 0
        if self.guard_y == 0:
            guard_action_mask[2] = 0
        elif self.guard_y == 6:
            guard_action_mask[3] = 0

        if self.guard_x - 1 == self.escape_x:
            guard_action_mask[0] = 0
        elif self.guard_x + 1 == self.escape_x:
            guard_action_mask[1] = 0
        if self.guard_y - 1 == self.escape_y:
            guard_action_mask[2] = 0
        elif self.guard_y + 1 == self.escape_y:
            guard_action_mask[3] = 0

        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        if self.prisoner_x == self.guard_x and self.prisoner_y == self.guard_y:
            rewards = {"prisoner": -1, "guard": 1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        elif self.prisoner_x == self.escape_x and self.prisoner_y == self.escape_y:
            rewards = {"prisoner": 1, "guard": -1}
            terminations = {a: True for a in self.agents}
            self.agents = []

        truncations = {"prisoner": False, "guard": False}
        if self.timestep > self.max_timesteps:
            rewards = {"prisoner": 0, "guard": 0}
            truncations = {"prisoner": True, "guard": True}
            self.agents = []
        self.timestep += 1

        observation = (
            self.prisoner_x + 7 * self.prisoner_y,
            self.guard_x + 7 * self.guard_y,
            self.escape_x + 7 * self.escape_y,
        )
        observations = {
            "prisoner": {
                "observation": observation,
                "action_mask": prisoner_action_mask,
            },
            "guard": {
                "observation": observation,
                "action_mask": guard_action_mask
            }
        }

        infos = {"prisoner": {}, "guard": {}}
        self.render(prisoner_prev_position, guard_prev_position)

        return observations, rewards, terminations, truncations, infos
    
    def render(self, prisoner_prev_position = None, guard_prev_position = None):
        if prisoner_prev_position is not None and (prisoner_prev_position[0] != self.escape_x or prisoner_prev_position[1] != self.escape_y):
            self.map[prisoner_prev_position[1]][prisoner_prev_position[0]][0] = 127
            self.map[prisoner_prev_position[1]][prisoner_prev_position[0]][1] = 0
            self.map[prisoner_prev_position[1]][prisoner_prev_position[0]][2] = 127
        if guard_prev_position is not None and (guard_prev_position[0] != self.escape_x or guard_prev_position[1] != self.escape_y):
            self.map[guard_prev_position[1]][guard_prev_position[0]][0] = 0
            self.map[guard_prev_position[1]][guard_prev_position[0]][1] = 127
            self.map[guard_prev_position[1]][guard_prev_position[0]][2] = 0
        if self.prisoner_x != self.escape_x and self.prisoner_y != self.escape_y:
            self.map[self.prisoner_y][self.prisoner_x][0] = 255
            self.map[self.prisoner_y][self.prisoner_x][1] = 0
            self.map[self.prisoner_y][self.prisoner_x][2] = 255
        if self.guard_x != self.escape_x and self.guard_y != self.escape_y:
            self.map[self.guard_y][self.guard_x][0] = 0
            self.map[self.guard_y][self.guard_x][1] = 255
            self.map[self.guard_y][self.guard_x][2] = 0
        imwrite(f"images/img_{self.timestep}.png", self.map)
        
    @functools.lru_cache(maxsize = None)
    def observation_space(self, agent) -> Space:
        return MultiDiscrete([7 * 7 - 1] * 3)
    
    @functools.lru_cache(maxsize = None)
    def action_space(self, agent) -> Space:
        return Discrete(4)