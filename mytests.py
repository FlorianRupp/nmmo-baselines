'''Documented at neuralmmo.github.io'''
import random
from pdb import set_trace as T

import nmmo
from nmmo import Terrain
import numpy as np
from random import shuffle

# Scripted models included with the baselines repository
from scripted import baselines


def simulate(env, config, render=False, horizon=float('inf')):
    '''Simulate an environment for a fixed horizon'''

    # Environment accepts a config object
    env = env(config())
    env.reset()

    t = 0
    while True:
        if render:
            env.render()

        # Scripted API computes actions
        obs, rewards, dones, infos = env.step({})

        # Later examples will use a fixed horizon
        t += 1
        if t >= horizon:
            break

    # Called at the end of simulation to obtain logs
    return env.terminal()


def gen_map(env, config, render=False, horizon=float('inf')):
    env = env(config())


class CustomMapGenerator(nmmo.MapGenerator):
    '''Subclass the base NMMO Map Generator'''

    def generate_map(self, idx):
        '''Override the default per-map generation method'''
        size = self.config.MAP_SIZE

        # Create fractal and material placeholders
        fractal = np.zeros((size, size))  # Unused in demo
        matl = np.array([nmmo.Terrain.WATER, nmmo.Terrain.TREE] * int(size / 2) * size).reshape(size, size)
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

        matl = np.random.choice(list(terrains.keys()), size=(center_size, center_size), p=list(terrains.values())).astype(np.uint8)
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


class Config(nmmo.config.Medium, nmmo.config.AllGameSystems):
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


if __name__ == '__main__':
    simulate(nmmo.Env, Config, render=True)
    gen_map(nmmo.Env, Config)
