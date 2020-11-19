import math
import logging

import numpy as np
from numpy import linalg as LA

from auxiliary_files.grid import Grid
from auxiliary_files.item import Item
from multi_scale_search import auxiliary_functions
from multi_scale_search.core import O_Cell
import config


# The World is the ground truth model, which the agent partially observes.
class WorldModel:
    # world gets map (grid) and agents initial configuration
    # note: world_grid is in general not the same as the grid the agent has
    # agent_conf0 is a tuple of x,y,theta
    def __init__(self, world_grid=Grid(1, 1, 1, 1), agent_conf0=(0, 0, 0)):
        self.log = logging.getLogger(__name__)
        # grid is of type Grid. The value is 'free if cell is empty, 'mug' if the mug is there, etc.
        self.grid = world_grid
        # agent/robot conf.
        self.agent_x = agent_conf0[0]
        self.agent_y = agent_conf0[1]
        self.agent_theta = agent_conf0[2]
        self.agent_carries = 'none'  # agent can only carry 1 item
        # cell the agent is in
        agent_cell = self.grid.get_cell_indices_by_position(self.agent_x, self.agent_y)
        self.agent_u = agent_cell[0]
        self.agent_v = agent_cell[1]
        # indicator for what the agent carries.
        # task = dictionary with keys = items and value = goal
        self.items = []
        self.item_agent_carries = 'none'
        self.task = {}
        self.open_tasks = {}
        self.finished_tasks = []  # list of item_names that are delivered
        self.time = 0

    # communicate task to world. task = dictionary with keys = items and value = goal
    def set_task(self, task):
        self.open_tasks = task
        self.task = task

    def add_item_to_task(self, item_type, item_x_goal, item_y_goal):
        self.task[item_type] = (item_x_goal, item_y_goal)
        self.open_tasks[item_type] = (item_x_goal, item_y_goal)

    def add_item_to_grid(self, item_type, item_x, item_y):
        item_u, item_v = self.grid.get_cell_indices_by_position(item_x, item_y)
        self.grid.set_cell_value(item_type, item_u, item_v)
        for item in self.items:
            if item.item_type == item_type:
                self.items.remove(item)
        self.items.append(Item(item_type=item_type, u=item_u, v=item_v, x=item_x, y=item_y))

    # this is NOT the same as the POMDP observation model. The world knows where the agent is and evaluates the agents
    # action and returns the observation (tuple: (list of cell-midpos & value, z_carry) the agent gets
    def evaluate_action_get_observation(self, action_value):
        # first world-transition model is applied to get new robot conf. and world conf.
        self.evaluate_transition_model(action_value)
        # now world-observation model is evaluated to get the agents observation
        agent_observation = self.evaluate_observation_model()

        return agent_observation

    # the world-transition model determines the next state and updates the time it took to get to that state
    def evaluate_transition_model(self, action_value):
        # action_value is either a pose (list [x, y, theta]) or special action (pickup/release)
        if type(action_value) != str:
            # calculate next position. to avoid rounding issues of cos and sin -> use round
            agent_x_new = action_value[0]
            agent_y_new = action_value[1]
            agent_theta_new = action_value[2]

            agent_x_old, agent_y_old, agent_theta_old = self.agent_x, self.agent_y, self.agent_theta
            # illegal moves are not possible
            if (0 < agent_x_new < self.grid.total_width) and (0 < agent_y_new < self.grid.total_height):
                # check that agent does not drive through the wall:
                # calculate new cell-value
                new_cell_u, new_cell_v = self.grid.get_cell_indices_by_position(agent_x_new, agent_y_new)
                if self.grid.cells[new_cell_v][new_cell_u].value == 'free':
                    self.agent_x = agent_x_new
                    self.agent_y = agent_y_new
                    self.agent_theta = agent_theta_new

                    self.agent_u = new_cell_u
                    self.agent_v = new_cell_v
                    # update x,y position of the item the agent carries
                    if self.agent_carries != 'none':
                        self.item_agent_carries.set_position(self.agent_x, self.agent_y)
                else:
                    msg = 'agent tries to drive into {} space'.format(self.grid.cells[new_cell_v][new_cell_u].value)
                    print(msg)
                    self.log.info(msg)
            else:
                print('agent tries to drive outside of the map')
                self.log.info('agent tries to drive outside of the map')
            self.update_world_time((agent_x_old, agent_y_old, agent_theta_old), action_value,
                                   state_new=(self.agent_x, self.agent_y, self.agent_theta))

        elif action_value == 'release_item' and self.agent_carries != 'none':
            goal = self.open_tasks[self.agent_carries]
            dist = (self.agent_x - goal[0]) ** 2 + (self.agent_y - goal[1]) ** 2
            # set item to new position
            item = self.item_agent_carries
            self.add_item_to_grid(item.item_type, item.x, item.y)
            if dist < config.robot_range ** 2:
                # if carrying an item from task and agent is close enough to goal location: subtask is finished
                if self.agent_carries in self.open_tasks:
                    self.finished_tasks.append(self.agent_carries)
                    del self.open_tasks[self.agent_carries]
            self.update_world_time(self.agent_carries, action_value, 'none')
            self.item_agent_carries = 'none'
            self.agent_carries = 'none'

        elif action_value[0:6] == 'pickup' and self.agent_carries == 'none':
            # pickup action
            item_name = action_value[7:]
            #  check all items in world
            item = self.items[0]
            for item in self.items:
                if item.item_type == item_name:
                    break
            # if the item is close enough: pickup succeeds
            dist = item.get_distance(self.agent_x, self.agent_y, self.grid.cell_width, self.grid.cell_height)
            is_pickup_successful = False
            if dist < config.robot_range:
                is_pickup_successful = True
                self.grid.set_cell_value('free', item.u, item.v)
            if is_pickup_successful:
                self.agent_carries = item_name
                self.item_agent_carries = item
            self.update_world_time(-1, action_value, self.agent_carries)

    # observations are status of cells observed (z_cells) and status of what robot carries (z_carry)
    # return type is tuple with values: (z_pos (3-tuple), z_cell = list of (cell-midpos, values), z_carry \in {0, 1, 2, 3})
    def evaluate_observation_model(self):
        o_agent_pose = (self.agent_x, self.agent_y, self.agent_theta)
        # list of cells observed, each list-entry is a tuple: (cell_midpos_x, cell_midpos_y, cell_value)
        o_cells = self.get_world_cells_observed()
        o_carry = self.agent_carries
        o_task = self.finished_tasks
        world_time = self.time

        return o_agent_pose, o_cells, o_carry, o_task, world_time

    # returns the cells of the world the agent currently observes as list with entries: (x, y, value)
    def get_world_cells_observed(self):
        # get transformation function of robot:
        trans_matrix = np.array([[math.cos(self.agent_theta), math.sin(self.agent_theta), self.agent_x],
                                 [-math.sin(self.agent_theta), math.cos(self.agent_theta), self.agent_y],
                                 [0, 0, 1]])
        # get transformed viewing cone points
        cells_dict = {}  # a dict is used to avoid adding a cell twice
        line_resolution = 0.025
        for ray in config.robot_viewing_cone:
            # get homogenous points of first and last one of line
            p_0h = np.array([ray[0][0], ray[0][1], 1])
            p_nh = np.array([ray[1][0], ray[1][1], 1])
            # get transformed points
            p_0trans = np.dot(trans_matrix, p_0h)
            p_ntrans = np.dot(trans_matrix, p_nh)
            # get unit vector pointing in line direction
            vec = np.array([p_ntrans[0] - p_0trans[0], p_ntrans[1] - p_0trans[1]])
            vec = 1. / (LA.norm(vec)) * vec
            for i in range(0, int(math.ceil(3 / line_resolution) + 1)):
                p_i = p_0trans[0:2] + i * line_resolution * vec
                u, v = self.grid.get_cell_indices_by_position(p_i[0], p_i[1])
                if u == -1 or v == -1:
                    break
                # if u,v is not already in cells_dict and is inside of world: add observation to cells_dict
                if (u, v) not in cells_dict.keys():
                    x, y = self.grid.get_position_by_indices(u, v)  # not x,y != p_i, since x,y is midpoint of cell
                    if self.grid.cells[v][u].value == 'occupied':
                        break
                    # check if an item is close, if so, change observation to item
                    obs_is_item = False
                    for item in self.items:
                        if -1 <= item.u - u <= 1 and -1 <= item.v - v <= 1:
                            for dv in range(-1, 2):
                                for du in range(-1, 2):
                                    if du == 0 and dv == 0:
                                        cells_dict[(item.u, item.v)] = \
                                            O_Cell(item.x, item.y, self.grid.cells[item.v][item.u].value)
                                    else:
                                        cells_dict[(item.u + du, item.v + dv)] = \
                                            O_Cell(x, y, self.grid.cells[item.v + dv][item.u + du].value)

                            obs_is_item = True
                    if not obs_is_item:
                        cells_dict[(u, v)] = O_Cell(x, y, self.grid.cells[v][u].value)
        # add agents position as observation, this prevents errors in some action subroutines
        cells_dict[(self.agent_u, self.agent_v)] = \
            O_Cell(self.agent_x, self.agent_y, self.grid.cells[self.agent_v][self.agent_u].value)
        # extracts value of cells_dict
        return list(cells_dict.values())

    # updates the world_time for a state, action, new_state triple
    def update_world_time(self, state, action_value, state_new):
        # if the robot successfully moved:
        if (type(action_value) != str) and state != state_new:
            # here it is assumed that the agent only rotated or only translated. no longer complicated maneuvers.
            self.time += math.sqrt((state_new[0] - state[0]) ** 2 + (state_new[1] - state[1]) ** 2) / config.robot_speed
            self.time += math.fabs(
                auxiliary_functions.angle_consistency(state_new[2] - state[2])) / config.robot_angular_speed
        # if the robot tries to move, but it does not work
        elif (type(action_value) != str) and state == state_new:
            self.time += 1  # does not move, but still wastes time for noticing that it does not work
        # if the robot releases item
        elif action_value == 'release_item':
            if state != state_new:
                self.time += config.robot_release_time
            else:
                self.time += 1  # penalty for failed release action
        elif action_value[0:6] == 'pickup':
            # if successful pickup
            if action_value[7:] == state_new:
                self.time += config.robot_pickup_time
            else:
                self.time += 1  # penalty for unsuccessful pickup
        else:
            print('Error: time not defined for input ({},{},{})'.format(state, action_value, state_new))
            self.log.warning('Error: time not defined for input ({},{},{})'.format(state, action_value, state_new))
