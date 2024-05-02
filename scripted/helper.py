import heapq

import numpy as np
from nmmo import Terrain
import nmmo

_ = nmmo.Env()

terrain_idx_mapping = {Terrain.GRASS: 2, Terrain.FOREST: 1, Terrain.STONE: 5, Terrain.WATER: 3, Terrain.SCRUB: 2, 0: 0}


def run_dikjstra(x, y, map, passable_values):
    print(map)
    dikjstra_map = np.full((len(map), len(map[0])), -1)
    visited_map = np.zeros((len(map), len(map[0])))
    queue = [(x, y, 0)]
    while len(queue) > 0:
        (cx, cy, cd) = queue.pop(0)
        cx, cy, cd = int(cx), int(cy), int(cd)
        if map[cy][cx] not in passable_values or (dikjstra_map[cy][cx] >= 0 and dikjstra_map[cy][cx] <= cd):
            continue
        visited_map[cy][cx] = 1
        dikjstra_map[cy][cx] = cd
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= len(map[0]) or ny >= len(map):
                continue
            queue.append((nx, ny, cd + 1))
    return dikjstra_map, visited_map


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f


def heuristic(node, goal):
    # Manhattan distance heuristic
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def is_valid_move(grid, x, y):
    # Check if the move is within the grid and the cell is not blocked
    x, y = int(x), int(y)
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] != 5 and grid[x][y] != 3


def astar(grid, start, goal):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    open_list = []
    closed_set = set()

    start_node = Node(start[0], start[1])
    goal_node = Node(goal[0], goal[1])

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add((current_node.x, current_node.y))

        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]  # Reverse the path

        for dx, dy in directions:
            next_x, next_y = current_node.x + dx, current_node.y + dy

            if not is_valid_move(grid, next_x, next_y) or (next_x, next_y) in closed_set:
                continue

            neighbor = Node(next_x, next_y, current_node)
            neighbor.g = current_node.g + 1
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h

            heapq.heappush(open_list, neighbor)

    return []  # No path found


def get_direction(pos1, pos2):
    diff = np.array(pos1) - np.array(pos2)
    if diff[0] == -1 and diff[1] == 0:
        return nmmo.action.South
    elif diff[0] == 1 and diff[1] == 0:
        return nmmo.action.North
    elif diff[0] == 0 and diff[1] == -1:
        return nmmo.action.East
    elif diff[0] == 0 and diff[1] == 1:
        return nmmo.action.West
    elif diff[0] == 0 and diff[1] == 0:
        return {}  # "DONT MOVE"
    else:
        raise ValueError


def format_map(obs):
    print(terrain_idx_mapping)
    # TODO fix observation size bug
    o = np.full([20, 20], 0)
    # l = []
    for row in obs:
        # l.append(row[2])
        o[int(row[2])][int(row[3])] = terrain_idx_mapping[int(row[1])]
    # print(max(l))
    return o[7:13, 7:13]


class NoFoodReachableException(Exception):
    def __init__(self):
        super().__init__("No food reachable.")


def get_closest_food(pos, o):
    dists = run_dikjstra(pos[0], pos[1], o, [1, 2])[0]
    # get all food distances
    ys, xs = np.where(o == 1)
    dists = [dists[y][x] for x, y in zip(xs, ys)]
    closest = np.argsort(dists)
    if len(closest) == 0:
        raise NoFoodReachableException()
    return ys[closest[0]], xs[closest[0]]
