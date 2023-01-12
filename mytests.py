'''Documented at neuralmmo.github.io'''
import random
from pdb import set_trace as T

import nmmo
from nmmo import Terrain
import numpy as np
from random import shuffle

# Scripted models included with the baselines repository
from scripted import baselines
from nmmo.lib.spawn import spawn_concurrent, spawn_continuous
import sys
import nmmo.core.realm


def simulate(env, config, render=False, horizon=float('inf')):
    '''Simulate an environment for a fixed horizon'''

    # Environment accepts a config object
    env = env(config())
    env.reset()

    t = 0
    while env.num_agents != 0:
        if render:
            env.render()

        # Scripted API computes actions
        obs, rewards, dones, infos = env.step({})

        # Later examples will use a fixed horizon
        t += 1
        if t >= horizon:
            break

    print("All agents died, stopping now")
    env.close()
    # Called at the end of simulation to obtain logs
    # return env.terminal()


def gen_map(env, config, render=False, horizon=float('inf')):
    env = env(config())


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


# class MyPlayerLoader(nmmo.lib.spawn):
#     pass

def spawn(config):
    return [(16, 16), (16, 18)]


class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
    '''Config objects subclass a nmmo.config.{Small, Medium, Large} template

    Can also specify config game systems to enable various features'''

    # Agents will be instantiated using templates included with the baselines
    # Meander: randomly wanders around
    # Forage: explicitly searches for food and water
    # Combat: forages and actively fights other agents
    SPECIALIZE = True

    PLAYERS = [baselines.Fisher]

    PLAYER_N = 2

    PLAYER_DEATH_FOG = None

    # Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    # Enable rendering
    RENDER = True

    # Force terrain generation -- avoids unexpected behavior from caching
    MAP_FORCE_GENERATION = True

    NPC_N = 0

    # MAP_N_TILE = 1
    MAP_GENERATE_PREVIEWS = True
    MAP_PREVIEW_DOWNSCALE = 1
    MAP_CENTER = 10
    # MAP_BORDER = 0
    MAP_GENERATOR = DummyMapGenerator
    LOG_VERBOSE = False
    LOG_EVENTS = False

    PROGRESSION_SYSTEM_ENABLED = False

    PLAYER_SPAWN_FUNCTION = spawn
    # PLAYER_SPAWN_FUNCTION = spawn_concurrent
    # TODO big pormblem is how to decide where agents spawn in a cumstom function
    # how to frame it competitive?


def printConfig(config):
    for attr in dir(config):
        if not attr.startswith('__'):
            print('{}: {}'.format(attr, getattr(config, attr)))


if __name__ == '__main__':
    # printConfig(Config())
    # print("myspawn", spawn(Config()))
    # print(spawn_concurrent(Config()))
    simulate(nmmo.Env, Config, render=True)
    # gen_map(nmmo.Env, Config)
