from cv2 import imread, imwrite

import functools
import random
from copy import copy, deepcopy

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
        self.agent_positions = None
        self.target_positions = None
        self.targets_achieved = None
        self.timestep = None
        self.max_timesteps = max_timesteps if max_timesteps is not None else 2000
        self.x_lim = self.map.shape[1] - 1
        self.y_lim = self.map.shape[0] - 1
        self.possible_agents = [f"agent_{i}" for i in range(4)]
        self.agent_colors = [[0, 0, 255], [0, 255, 0], [255, 0, 255], [255, 255, 0]]

    def reset(self, seed = None, options = None):
        self.agents = copy(self.possible_agents)
        self.target_positions = []
        self.timestep = 0
        self.targets_achieved = []
        self.agent_positions = [[0, 0], [0, self.y_lim], [self.x_lim, 0], [self.x_lim, self.y_lim]]

        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if self.map[y][x][0] == 255 and self.map[y][x][1] == 0 and self.map[y][x][2] == 0:
                    self.target_positions.append([x, y])
                    self.targets_achieved.append(0)
        
        observations = {
            agent : {
                "observation" : obs[0] + 7 * obs[1],
                "action_mask" : [obs[0] == 0, obs[0] == self.x_lim - 1, obs[1] == 0, obs[1] == self.y_lim - 1]
            }
            for agent, obs, in zip(self.agents, self.target_positions)
        }

        infos = {
            a : {} for a in self.agents
        }
        self.render()
        
        return observations, infos
    
    def step(self, actions):
        prev_positions = deepcopy(self.agent_positions)
        
        for i, agent in enumerate(self.agents):
            action = actions[agent]
            if action == 0 and self.agent_positions[i][0] > 0:
                if (
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] - 1][0] != 0 or
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] - 1][1] != 0 or
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] - 1][2] != 0
                ) and (
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] - 1].tolist() not in self.agent_colors
                ):
                    self.agent_positions[i][0] -= 1
            if action == 1 and self.agent_positions[i][0] < self.x_lim:
                if (
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] + 1][0] != 0 or
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] + 1][1] != 0 or
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] + 1][2] != 0
                ) and (
                    self.map[self.agent_positions[i][1]][self.agent_positions[i][0] + 1].tolist() not in self.agent_colors
                ):
                    self.agent_positions[i][0] += 1
            if action == 2 and self.agent_positions[i][1] > 0:
                if (
                    self.map[self.agent_positions[i][1] - 1][self.agent_positions[i][0]][0] != 0 or
                    self.map[self.agent_positions[i][1] - 1][self.agent_positions[i][0]][1] != 0 or
                    self.map[self.agent_positions[i][1] - 1][self.agent_positions[i][0]][2] != 0
                ) and (
                    self.map[self.agent_positions[i][1] - 1][self.agent_positions[i][0]].tolist() not in self.agent_colors
                ):
                    self.agent_positions[i][1] -= 1
            if action == 3 and self.agent_positions[i][1] < self.y_lim:
                if (
                    self.map[self.agent_positions[i][1] + 1][self.agent_positions[i][0]][0] != 0 or
                    self.map[self.agent_positions[i][1] + 1][self.agent_positions[i][0]][1] != 0 or
                    self.map[self.agent_positions[i][1] + 1][self.agent_positions[i][0]][2] != 0
                ) and (
                    self.map[self.agent_positions[i][1] + 1][self.agent_positions[i][0]].tolist() not in self.agent_colors
                ):
                    self.agent_positions[i][1] += 1

        rewards = {a: 0 for a in self.agents}
        for agent, position in zip(self.agents, self.agent_positions):
            if position in self.target_positions:
                ind = self.target_positions.index(position)
                if self.targets_achieved[ind] == 0:
                    self.targets_achieved[ind] = 1
                    rewards[agent] += 1
        terminations = {
            agent : sum(self.targets_achieved) == len(self.targets_achieved) for agent in self.agents
        }
        
        if sum(self.targets_achieved) == len(self.targets_achieved):
            self.agents = []

        truncations = {
            agent : False for agent in self.agents
        }
        if self.timestep > self.max_timesteps:
            rewards = {
                agent : 0 for agent in self.agents
            }
            truncations = {
                agent : True for agent in self.agents
            }
            self.agents = []
        self.timestep += 1

        observations = {
            agent : {
                "observation" : obs[0] + 7 * obs[1],
                "action_mask" : [obs[0] == 0, obs[0] == self.x_lim - 1, obs[1] == 0, obs[1] == self.y_lim - 1]
            }
            for agent, obs, in zip(self.agents, self.target_positions)
        }

        infos = {
            a : {} for a in self.agents
        }
        self.render(prev_positions)

        return observations, rewards, terminations, truncations, infos
    
    def render(self, prev_positions = None):
        if prev_positions is not None:
            for i, (prev_position, position) in enumerate(zip(prev_positions, self.agent_positions)):
                self.map[prev_position[1]][prev_position[0]] = [self.agent_colors[i][0] // 2, self.agent_colors[i][1] // 2, self.agent_colors[i][2] // 2]
                self.map[position[1]][position[0]] = self.agent_colors[i]
        else:
            for i, position in enumerate(self.agent_positions):
                self.map[position[1]][position[0]] = self.agent_colors[i]
        imwrite(f"images/img_{self.timestep}.png", self.map)
        
    @functools.lru_cache(maxsize = None)
    def observation_space(self, agent) -> Space:
        return MultiDiscrete([7 * 7 - 1] * 3)
    
    @functools.lru_cache(maxsize = None)
    def action_space(self, agent) -> Space:
        return Discrete(4)