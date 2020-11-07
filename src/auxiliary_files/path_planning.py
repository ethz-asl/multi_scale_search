# This file implements the A* algorithm
# This code is copied from  https://gist.github.com/jamiees2/5531924 and modified by myself
import math
import time
import logging

import numpy as np

import config


class Node:
    # value in {free, occupied}, point = (u,v)
    def __init__(self, u, v, theta=0.0):
        self.u = u
        self.v = v
        self.theta = theta
        self.parent = None
        self.H = 0
        self.G = 0


def astar(start_u, start_v, theta0, goal_u, goal_v, grid):
    start_time = time.time()
    # initialize open and closed set
    open_dict = {}  # key = point(=(u,v)), value = node object
    closed_dict = {}
    # set start node
    current = Node(u=start_u, v=start_v, theta=theta0)
    open_dict[(start_u, start_v)] = current

    # check if goal cell occupied
    if grid.cells[goal_v][goal_u].value == 'occupied':
        # set neighbour cell as target instead
        found_valid_neighbour = False
        for dv in range(-1, 2):
            for du in range(-1, 2):
                if 0 <= goal_u + du < len(grid.cells[0]) and 0 <= goal_v + dv < len(grid.cells) and \
                        grid.cells[goal_u + dv][goal_v + du].value == 'free':
                    goal_u, goal_v = goal_u + du, goal_v + dv
                    found_valid_neighbour = True
                    break
            if found_valid_neighbour:
                break
    # while open-set not empty
    while len(open_dict) > 0:
        # get node in open_dict with lowest G + H score
        current = min(open_dict.values(), key=lambda o: o.G + o.H)
        if current.u == goal_u and current.v == goal_v:
            path = []
            final_g = current.G
            while current.parent:
                path.append((current.u, current.v))
                current = current.parent
            # path.append(current.point)  # only needed if
            end_time = time.time()
            delta_time = end_time - start_time
            return path[::-1], final_g  # path vector is from goal to start, revert it before returning
        # remove current node from open set, add to closed set
        del open_dict[(current.u, current.v)]
        closed_dict[(current.u, current.v)] = current
        # loop through node's neighbours
        neighbour_nodes = get_neighbours(current, grid.cells)
        for node_key in neighbour_nodes:
            new_g = current.G + transition_cost(current, node_key, grid.cell_width, grid.cell_height)
            new_theta = 2 * math.pi - math.atan2(node_key[1] - current.v, node_key[0] - current.u)
            # if node already in closed set
            if node_key in closed_dict.keys():
                if new_g < closed_dict[node_key].G:
                    # remove node form closed_dict and add it to open_dict again
                    node = closed_dict[node_key]
                    del closed_dict[node_key]
                    node.G = new_g
                    node.theta = new_theta
                    open_dict[node_key] = node
            # if already in open_dict
            elif node_key in open_dict:
                # check if we beat the G score
                if new_g < open_dict[node_key].G:
                    # update cost, theta and parent of node
                    open_dict[node_key].G = new_g
                    open_dict[node_key].theta = new_theta
                    open_dict[node_key].parent = current
            # if not in open_dict and not in closed dict:
            else:
                # add new node to open_dict
                open_dict[node_key] = Node(u=node_key[0], v=node_key[1], theta=new_theta)
                # calculate G and H score for the node and add to open_dict
                open_dict[node_key].G = new_g
                open_dict[node_key].H = heuristic(open_dict[node_key], goal_u, goal_v, grid.cell_width,
                                                  grid.cell_height)
                # set parent to current node
                open_dict[node_key].parent = current
    # Throw an exception if there is no path
    raise ValueError('No Path Found, start_u={}, start_v={}, theta0={}, goal_u={}, goal_v={}'.format(
        start_u, start_v, theta0, goal_u, goal_v))


def get_neighbours(node, grid_cells):
    u0, v0, theta0 = node.u, node.v, node.theta
    neighbours = set()
    # get grid-cell right in front of agent
    for dv in range(-1, 2):
        for du in range(-1, 2):
            if du == 0 and dv == 0:
                continue
            if 0 <= u0 + du < len(grid_cells[0]) and 0 <= v0 + dv < len(grid_cells) and \
                    grid_cells[v0 + dv][u0 + du].value == 'free':
                neighbours.add((u0 + du, v0 + dv))
    return neighbours


def transition_cost(node_i, node_j_key, cell_width, cell_height):
    # there is a cost associated to rotating towards node_j and a cost for driving to node_j
    cost = 0
    # 2pi - atan2 because atan2 is for a normal coordinate system where y-axis points upwards.
    theta_ij = 2 * math.pi - math.atan2(node_j_key[1] - node_i.v, node_j_key[0] - node_i.u)
    delta_theta = math.fabs(node_i.theta - theta_ij) % (2 * math.pi)
    delta_theta = min([delta_theta, 2 * math.pi - delta_theta])
    cost += delta_theta / config.robot_angular_speed
    # cost for driving to node_j:
    dist = math.sqrt((cell_width * (node_j_key[0] - node_i.u)) ** 2 +
                     (cell_height * (node_j_key[1] - node_i.v)) ** 2)
    cost += dist / config.robot_speed
    return cost


def heuristic(node, goal_u, goal_v, cell_width, cell_height):
    cost = 0
    # cost of rotating towards goal-cell
    # 2pi - atan2 because atan2 is for a normal coordinate system where y-axis points upwards.
    theta_ij = 2 * math.pi - math.atan2(goal_v - node.v, goal_u - node.u)
    delta_theta = math.fabs(node.theta - theta_ij) % (2 * math.pi)
    delta_theta = min([delta_theta, 2 * math.pi - delta_theta])
    cost += delta_theta / config.robot_angular_speed
    # cost of driving to cost-cell
    dist = math.sqrt((cell_width * (node.u - goal_u)) ** 2 + (cell_height * (node.v - goal_v)) ** 2)
    cost += dist / config.robot_speed

    return cost


def get_neighbours_theta(node, grid_cells):
    u0, v0, theta0 = node.u, node.v, node.theta
    neighbours = set()
    # get grid-cell right in front of agent
    du = int(round(math.cos(theta0)))
    dv = int(round(math.sin(-theta0)))
    if 0 <= u0 + du < len(grid_cells[0]) and 0 <= v0 + dv < len(grid_cells) and \
            grid_cells[v0 + dv][u0 + du].value == 'free':
        neighbours.add((u0 + du, v0 + dv, theta0))
    # get the nodes in the same cell but with different angle
    dtheta_values = np.arange(math.pi / 4, 2 * math.pi - 0.001, math.pi / 4)
    for dtheta in dtheta_values:
        neighbours.add((u0, v0, (theta0 + dtheta) % (2 * math.pi)))
    return neighbours


def transition_cost_theta(node_i, node_j_key, cell_width, cell_height):
    # if node_j_key is in same cell as agent the cost is for rotating
    cost = 0
    if node_i.theta != node_j_key[2] and node_i.u == node_j_key[0] and node_i.v == node_j_key[1]:
        theta_ij = math.fabs(node_j_key[2] - node_i.theta) % (2 * math.pi)
        theta_ij = min([theta_ij, 2 * math.pi - theta_ij])
        cost = theta_ij / config.robot_angular_speed
    elif node_i.theta == node_j_key[2] and (node_i.u != node_j_key[0] or node_i.v != node_j_key[1]):
        dist = math.sqrt((cell_width * (node_j_key[0] - node_i.u)) ** 2 +
                         (cell_height * (node_j_key[1] - node_i.v)) ** 2)
        cost = dist / config.robot_speed
    else:
        print('unexpected node_j_key')
        log = logging.getLogger(__name__)
        log.warning('unexpected node_j_key')
    return cost


def heuristic_theta(node, goal_key, cell_width, cell_height):
    # cost of rotating towards goal-cell
    # 2pi - atan2 because atan2 is for a normal coordinate system where y-axis points upwards.
    theta_ij = 2 * math.pi - math.atan2(goal_key[1] - node.v, goal_key[0] - node.u)
    delta_theta = math.fabs(node.theta - theta_ij) % (2 * math.pi)
    delta_theta = min([delta_theta, 2 * math.pi - delta_theta])
    cost_theta = delta_theta / config.robot_angular_speed
    # cost of driving to cost-cell
    dist = math.sqrt((cell_width * (node.u - goal_key[0])) ** 2 + (cell_height * (node.v - goal_key[1])) ** 2)
    cost_dist = dist / config.robot_speed
    # cost of rotating to final angle
    if goal_key[2] == 'any':
        cost_theta2 = 0
    else:
        delta_theta2 = math.fabs(goal_key[2] - theta_ij) % (2 * math.pi)
        delta_theta2 = min([delta_theta2, 2 * math.pi - delta_theta2])
        cost_theta2 = delta_theta2 / config.robot_angular_speed

    return cost_theta + cost_dist + cost_theta2
