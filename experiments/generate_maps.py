import shutil

import nmmo
import pandas as pd
from tqdm import tqdm

from mytests import DummyMapGenerator, spawn
from scripted import baselines


class Config(nmmo.config.Small, nmmo.config.AllGameSystems):
    SPECIALIZE = True
    COMBAT_SYSTEM_ENABLED = True
    # PLAYERS = [EastAgent, WestAgent]
    PLAYERS = [baselines.Forage, baselines.Forage]

    PLAYER_N = 2
    PLAYER_DEATH_FOG = None

    # Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'
    MAP_N = 1
    RENDER = False
    MAP_FORCE_GENERATION = True

    NPC_N = 0
    MAP_CENTER = 8
    MAP_BORDER = 6
    MAP_GENERATOR = DummyMapGenerator

    LOG_VERBOSE = False
    LOG_EVENTS = False

    PROGRESSION_SYSTEM_ENABLED = False
    MAP_GENERATE_PREVIEWS = True

    PLAYER_SPAWN_FUNCTION = spawn


config = Config()


def simulate(environment, config, render=False, runs=1, delay=0):
    winners = []
    env = environment(config())
    for i in range(runs):
        env.reset()
        t = 0
        while True:
            if render:
                env.render()

            obs, rewards, dones, infos = env.step({})
            t += 1
            if env.num_agents <= 1:
                if env.num_agents == 1:
                    winners.append(env.agents[0])
                else:
                    winners.extend([1, 2])
                break
    env.close()
    return winners


def test_balancing(runs=1):
    winners = simulate(nmmo.Env, Config, runs=runs)
    balance = round(sum(winners) / len(winners), 2)
    print("Balance", balance)
    return balance


balancings = []
for i in tqdm(range(3000)):
    print(f"Generated map {i}")
    balancings.append({"idx": i, "balancing": test_balancing(runs=20)})
    shutil.copytree(r"maps/demos/map1", f"gen_maps/map{i}")


pd.DataFrame(balancings).to_csv("data.csv")
