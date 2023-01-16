'''Documented at neuralmmo.github.io'''
import random
from pdb import set_trace as T

import nmmo
from nmmo import Terrain
import numpy as np
import tqdm
from random import shuffle

# Scripted models included with the baselines repository
from pyasn1_modules.rfc3279 import Curve

from scripted import baselines
from nmmo.lib.spawn import spawn_concurrent, spawn_continuous
import sys
import nmmo.core.realm
from time import sleep


def simulate(environment, config, render=False, horizon=float('inf'), runs=1, delay=0):
    winners = []

    env = environment(config())
    for i in range(runs):
        # Environment accepts a config object
        env.reset()

        t = 0
        while True:
            # while env.num_agents != 0:
            if render:
                env.render()

            # Scripted API computes actions
            obs, rewards, dones, infos = env.step({})

            # Later examples will use a fixed horizon
            t += 1
            sleep(delay)
            if env.num_agents <= 1:
                if env.num_agents == 1:
                    winners.append(env.agents[0])
                else:
                    winners.extend([1, 2])
                break

    env.close()
    return winners


def gen_map(config):
    DummyMapGenerator(config).generate_map(1)


class CustomMapGenerator(nmmo.MapGenerator):
    '''Subclass the base NMMO Map Generator'''

    def generate_map(self, idx):
        '''Override the default per-map generation method'''
        size = self.config.MAP_SIZE

        # Create fractal and material placeholders
        fractal = np.zeros((size, size))  # Unused in demo
        # matl = np.array([nmmo.Terrain.WATER, nmmo.Terrain.TREE] * int(size / 2) * size).reshape(size, size)
        matl = np.array([nmmo.Terrain.GRASS] * size * size).reshape(size, size)
        # Return signature includes fractal and material
        # Pass a zero array if fractal is not relevant
        print(matl.shape)
        return fractal, matl


class DummyMapGenerator(nmmo.MapGenerator):
    # only works with size 6
    def generate_map(self, idx):
        size = self.config.MAP_SIZE
        center_size = self.config.MAP_CENTER

        fractal = np.zeros((size, size))

        terrains = {Terrain.STONE: 0.2, Terrain.GRASS: 0.4, Terrain.FOREST: 0.1, Terrain.TREE: 0.1, Terrain.ORE: 0.025,
                    Terrain.CRYSTAL: 0.025, Terrain.WATER: 0.15}

        matl = np.random.choice(list(terrains.keys()), size=(center_size, center_size),
                                p=list(terrains.values())).astype(np.uint8)
        matl[0] = [Terrain.STONE] * center_size
        matl[-1] = [Terrain.STONE] * center_size
        matl[:, 0] = [Terrain.STONE] * center_size
        matl[:, -1] = [Terrain.STONE] * center_size

        # make sure in 2 corners is always grass to be vacant to spawn
        player_pos = [(1, 1), (center_size - 2, center_size - 2)]
        matl[player_pos[0]] = Terrain.GRASS
        matl[player_pos[1]] = Terrain.GRASS

        matl = np.pad(matl, pad_width=self.config.MAP_BORDER)
        return fractal, matl


class CustomMapGenerator2(nmmo.MapGenerator):
    '''Subclass the base NMMO Map Generator'''

    def generate_map(self, idx):
        '''Override the default per-map generation method'''
        config = self.config
        size = config.MAP_SIZE

        # Create fractal and material placeholders
        fractal = np.zeros((size, size))  # Unused in demo
        matl = np.zeros((size, size), dtype=object)

        for r in range(size):
            for c in range(size):
                linf = max(abs(r - size // 2), abs(c - size // 2))

                # Set per-tile materials
                if linf < 4:
                    matl[r, c] = nmmo.Terrain.STONE
                elif linf < 8:
                    matl[r, c] = nmmo.Terrain.WATER
                elif linf < 12:
                    matl[r, c] = nmmo.Terrain.FOREST
                elif linf <= size // 2 - config.MAP_BORDER:
                    matl[r, c] = nmmo.Terrain.GRASS
                else:
                    matl[r, c] = nmmo.Terrain.LAVA

        # Return signature includes fractal and material
        # Pass a zero array if fractal is not relevant
        print(matl.shape)
        return fractal, matl


def spawn(config):
    # spawn agents in corners
    player1 = (config.MAP_BORDER + 1, config.MAP_BORDER + 1)
    player2 = (config.MAP_BORDER + config.MAP_CENTER - 2, config.MAP_BORDER + config.MAP_CENTER - 2)
    return [player1, player2]


class StandingAgent(nmmo.Agent):
    scripted = True

    def __call__(self, *args, **kwargs):
        return self.actions


class EastAgent(nmmo.Agent):
    scripted = True

    def __call__(self, *args, **kwargs):
        return {nmmo.action.Move: {nmmo.action.Direction: nmmo.action.East}}


class WestAgent(nmmo.Agent):
    scripted = True

    def __call__(self, *args, **kwargs):
        return {nmmo.action.Move: {nmmo.action.Direction: nmmo.action.West}}


RENDERING = False


class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
    '''Config objects subclass a nmmo.config.{Small, Medium, Large} template

    Can also specify config game systems to enable various features'''

    # Agents will be instantiated using templates included with the baselines
    # Meander: randomly wanders around
    # Forage: explicitly searches for food and water
    # Combat: forages and actively fights other agents
    SPECIALIZE = True
    COMBAT_SYSTEM_ENABLED = True
    # PLAYERS = [EastAgent, WestAgent]
    PLAYERS = [baselines.Forage, baselines.Forage]

    PLAYER_N = 2
    PLAYER_DEATH_FOG = None

    # Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'
    RENDER = RENDERING

    # Force terrain generation -- avoids unexpected behavior from caching
    MAP_FORCE_GENERATION = True

    NPC_N = 0
    MAP_GENERATE_PREVIEWS = False
    # MAP_PREVIEW_DOWNSCALE = 1
    MAP_CENTER = 16
    MAP_BORDER = 6
    # MAP_GENERATOR = CustomMapGenerator
    MAP_GENERATOR = DummyMapGenerator

    LOG_VERBOSE = False
    LOG_EVENTS = False

    PROGRESSION_SYSTEM_ENABLED = False

    PLAYER_SPAWN_FUNCTION = spawn


def print_config(config):
    for attr in dir(config):
        if not attr.startswith('__'):
            print('{}: {}'.format(attr, getattr(config, attr)))


if __name__ == '__main__':
    # printConfig(Config())
    over_all = []
    for i in tqdm.tqdm(range(1)):
        winners = simulate(nmmo.Env, Config, render=RENDERING, runs=20, delay=0)
        over_all.append(sum(winners)/len(winners))
    # gen_map(Config())
    print(over_all)