import math
import logging

import numpy as np
from numpy import linalg as LA

import config
from auxiliary_files.grid import Grid
from multi_scale_search import auxiliary_functions


class Action:
    # IMPORTANT NOTE: pickup actions need to start with pickup0 for the first item
    def __init__(self, name, file_name, layer, environment, xa_values, nodegraph):
        self.log = logging.getLogger(__name__)
        self.name = name
        self.file_name = file_name
        self.models_path = config.BASE_FOLDER_SIM + environment + '_environment/models/'
        self.file_name_t_expected = self.models_path + 't_expected_{}'.format(file_name)
        self.file_name_obs_prob = self.models_path + 'obs_prob_{}'.format(file_name)
        self.layer = layer
        self.environment = environment
        self.t_expected = []  # list with t_expected for each value of xa
        self.b0_tot = 0  # needed for POMDP
        self.start_max_belief = 0
        self.initialize_t_expected(xa_values, nodegraph)
        self.initial_call = True
        # self.reference = []
        self.goal_ref = [-1.0, -1.0, -1.0]

        self.start_xa = -1
        self.terminal_states = []
        self.steps_since_replanning = 0
        self.observation_probabilities = {}  # key is a specification to find the value
        # self.start_time = 0

    # overwritten by child class
    def initialize_t_expected(self, xa_values, nodegraph):
        return 'not initialized, gets overwritten by child class'

    def subroutine(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        if self.initial_call:
            self.initial_call_function(s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0)
        else:
            action_is_finished, item_present = self.is_action_finished(s, item_map, b0)
            if action_is_finished:
                self.finish_action(s, item_present=item_present)
                return 'finished'
            elif not self.is_goalref_still_valid(s, agent_x, agent_y, agent_theta, grid, b0):
                self.log.info('replanning')
                self.steps_since_replanning = 0
                self.goal_ref = self.core_algorithm(s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items,
                                                    b0)
        self.steps_since_replanning += 1
        return self.goal_ref

    # overwritten by some child classes
    def initial_call_function(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        self.start_xa = s['xa']
        self.goal_ref = self.core_algorithm(s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0)
        self.initial_call = False
        if b0 != -1:
            self.b0_tot = b0.get_aggregated_belief(node_nr=self.start_xa)
            self.start_max_belief = b0.get_max_belief_cell_value(self.start_xa)

    # overwritten by child class
    def core_algorithm(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        target_state = [-1.0, -1.0, -1.0]
        self.goal_ref = target_state
        return self.goal_ref

    def finish_action(self, s, item_present):
        self.initial_call = True
        self.goal_ref = [-1.0, -1.0, -1.0]
        self.steps_since_replanning = 0

    # overwritten by child class
    def is_goalref_still_valid(self, s, agent_x, agent_y, agent_theta, grid, b0=-1):
        return True

    # overwritten by child class
    # TODO: When adding a new simulation type (e.g. ROS) this method needs to be adapted for each action child-class
    def is_action_finished(self, s, item_map, b0):
        return False, False


def transform_path_to_states(agent_x, agent_y, agent_theta, grid, path, goal_theta='none'):
    agent_u, agent_v = grid.get_cell_indices_by_position(agent_x, agent_y)
    path_states = []
    for p in range(0, len(path)):
        p_u, p_v = path[p][0], path[p][1]
        p_x, p_y = grid.get_position_by_indices(p_u, p_v)
        # if in correct cell: get next point
        if agent_u == p_u and agent_v == p_v:
            continue
        theta = auxiliary_functions.angle_consistency(2 * math.pi - math.atan2(p_v - agent_v, p_u - agent_u))
        dtheta = auxiliary_functions.angle_consistency(theta - agent_theta)
        if not (-0.00001 < dtheta < 0.00001):
            path_states.append((agent_x, agent_y, theta))
        path_states.append((p_x, p_y, theta))
        agent_u, agent_v, agent_theta = p_u, p_v, theta
    if agent_theta != goal_theta and goal_theta != 'none':
        path_states.append((path_states[-1][0], path_states[-1][1], goal_theta))
    return path_states


def get_closest_free_cell(agent_u, agent_v, item_u, item_v, grid, max_item_goal_dist=0.8,
                          required_item_goal_dist=0.0):
    min_ag_goal_dist = 1000
    agent_x, agent_y = grid.get_position_by_indices(agent_u, agent_v)
    item_x, item_y = grid.get_position_by_indices(item_u, item_v)
    goal_u, goal_v = -1, -1
    for n in range(1, 5):
        for dv in range(-n, n + 1):
            for du in range(-n, n + 1):
                if 0 <= item_u + du < grid.nr_of_cells_x and 0 <= item_v + dv < grid.nr_of_cells_y and \
                        grid.cells[item_v + dv][item_u + du].value == 'free':
                    c_x, c_y = grid.get_position_by_indices(item_u + du, item_v + dv)
                    dist_ag_cell = math.sqrt((agent_x - c_x) ** 2 + (agent_y - c_y) ** 2)
                    dist_cell_item = math.sqrt((item_x - c_x) ** 2 + (item_y - c_y) ** 2)
                    if dist_ag_cell < min_ag_goal_dist and dist_cell_item >= required_item_goal_dist and \
                            dist_cell_item < max_item_goal_dist:
                        min_ag_goal_dist = dist_ag_cell
                        goal_u = item_u + du
                        goal_v = item_v + dv
        if min_ag_goal_dist < 1000:
            break
    return goal_u, goal_v


class Navigate(Action):
    def __init__(self, name, file_name, layer, environment, xa_values, nodegraph,
                 grid=Grid(nr_of_cells_x=1, nr_of_cells_y=1, world_width=1, world_height=1)):
        self.n_i = auxiliary_functions.nav_action_get_n0(name)
        self.n_j = auxiliary_functions.nav_action_get_n1(name)
        self.recs_i = nodegraph.node_recs[self.n_i]
        self.recs_j = nodegraph.node_recs[self.n_j]
        Action.__init__(self, name, file_name, layer, environment, xa_values, nodegraph)
        self.initialize_obs_probability(xa_values, nodegraph, grid=grid)
        self.log = logging.getLogger(__name__)

    def initialize_t_expected(self, xa_values, nodegraph):
        # read from file
        with open(self.file_name_t_expected) as f:
            time_list = f.readline().strip()
            self.t_expected = [float(val) for val in time_list.split(' ')]
        return

    def initialize_obs_probability(self, xa_values, nodegraph, grid):
        # read from file
        with open(self.file_name_obs_prob) as f:
            lines = [line.strip() for line in f]
        for obs_prob_values in lines:
            key = int(obs_prob_values.partition(':')[0])
            self.observation_probabilities[key] = [float(val) for val in obs_prob_values.partition(':')[2].split(' ')]
        return

    def core_algorithm(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        xa = s['xa']
        agent_u, agent_v = grid.get_cell_indices_by_position(agent_x, agent_y)
        # check if requirements for this action is met
        if xa != self.n_i and xa != self.n_j:
            return 'xa = {}, requirements for subroutine nav({},{}) are not met'.format(xa, self.n_i, self.n_j)
        # which is the goal node?:
        n_g = self.n_i
        if xa == self.n_i:
            n_g = self.n_j
        self.terminal_states = [n_g]
        goal_x, goal_y = nodegraph.get_center_of_gravity(node_nr=n_g)
        goal_ref = [goal_x, goal_y, 'none']

        return goal_ref

    def is_action_finished(self, s, item_map, b0):
        xa = s['xa']
        # criteria 1: if the max_belief increased, finish
        if config.simulation_name == 'low_fidelity_simulation':
            if b0 != -1 and self.start_max_belief > 0:
                b_max = b0.get_max_belief_cell_value(xa)
                if (b_max / self.start_max_belief) > (max(1.05, 0.9 / self.start_max_belief)):
                    self.log.info(self.name + 'finished, max belief over 0.9')
                    return True, True
        # criteria 2: if agent arrived in goal location
        if xa in self.terminal_states:
            return True, False
        else:
            return False, False


class Pickup(Action):
    def __init__(self, name, file_name, layer, file_name_look_around, environment, xa_values, nodegraph):
        self.item_i = auxiliary_functions.pickup_get_i(name)
        self.models_path = config.BASE_FOLDER_SIM + environment + '_environment/models/'
        self.file_name_look_around_t_expected = self.models_path + 't_expected_{}'.format(file_name_look_around)
        Action.__init__(self, name, file_name, layer, environment, xa_values, nodegraph)
        self.log = logging.getLogger(__name__)

    def initialize_t_expected(self, xa_values, nodegraph):
        # read t_expected(pickup if item is present) from file
        with open(self.file_name_t_expected) as f:
            times = [line.strip() for line in f]
        for time_list in times:
            self.t_expected.append([float(val) for val in time_list.split(' ')])
        # calculate t_expected(pickup if item is not present) from look_around times and pickup if item is present times
        with open(self.file_name_look_around_t_expected) as f:
            times_la = [line.strip() for line in f]
        t_expected_la = []
        for times_la_list in times_la:
            t_expected_la.append([float(val) for val in times_la_list.split(' ')])
        self.t_expected.insert(0, list(
            0.6 / (1 - 0.6) * np.array(t_expected_la[1]) + np.array(t_expected_la[0]) + np.array(self.t_expected[0])))
        return

    def initialize_grids(self, xa, nodegraph, cell_width, cell_height):
        # create a local grid
        min_x0 = min([rec.x0 for rec in nodegraph.node_recs[xa]])
        max_x1 = max([rec.x0 + rec.width for rec in nodegraph.node_recs[xa]])
        min_y0 = min([rec.y0 for rec in nodegraph.node_recs[xa]])
        max_y1 = max([rec.y0 + rec.height for rec in nodegraph.node_recs[xa]])

        nr_of_cells_x = int((max_x1 - min_x0) / cell_width)
        nr_of_cells_y = int((max_y1 - min_y0) / cell_height)
        grid = Grid(nr_of_cells_x, nr_of_cells_y, world_width=max_x1 - min_x0, world_height=max_y1 - min_y0,
                    x0=min_x0, y0=min_y0, default_value='occupied')
        # set 'free' regions
        for rec_idx, rec in enumerate(nodegraph.node_recs[xa]):
            # grid.set_region(rec, 'free')
            if self.environment == 'small':
                if self.layer == 1:
                    if xa == 2:
                        grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='smaller', y0_beh='smaller',
                                        y1_beh='bigger')
                    elif xa == 4:
                        grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='bigger', y0_beh='smaller',
                                        y1_beh='smaller')
                    else:
                        grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='smaller', y0_beh='smaller',
                                        y1_beh='smaller')
                elif self.layer == 2:
                    if xa == 8:
                        grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='smaller', y0_beh='smaller',
                                        y1_beh='bigger')
                    elif xa == 11 or xa == 14:
                        grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='bigger', y0_beh='smaller',
                                        y1_beh='smaller')
                    else:
                        grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='smaller', y0_beh='smaller',
                                        y1_beh='smaller')
            elif self.environment == 'big':
                if self.layer == 2:
                    if rec_idx == 0 and (xa == 1 or xa == 4 or xa == 7 or xa == 10):
                        grid.set_region(rec, 'free', y1_beh='bigger')
                    elif xa == 13 or xa == 16 or xa == 19 or xa == 22:
                        grid.set_region(rec, 'free', y1_beh='bigger')
                    else:
                        grid.set_region(rec, 'free')
        return grid

    def core_algorithm(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        agent_u, agent_v = grid.get_cell_indices_by_position(agent_x, agent_y)
        item = items[self.item_i]
        # get cell in belief grid with highest belief
        bmax_x, bmax_y = b0.get_max_belief_cell_pos(node_nr=b0.get_node_nr(agent_x, agent_y), item_nr=self.item_i)
        dist = math.sqrt((agent_x - bmax_x) ** 2 + (agent_y - bmax_y) ** 2)
        theta_ref = auxiliary_functions.angle_consistency(math.atan2(bmax_y - agent_y, bmax_x - agent_x))
        delta_theta = math.fabs(auxiliary_functions.angle_consistency(theta_ref - agent_theta))
        if dist < config.robot_range - 0.1 and delta_theta < 0.2:
            goal_ref = 'pickup_' + item.name
            return goal_ref

        gridmax_u, gridmax_v = grid.get_cell_indices_by_position(bmax_x, bmax_y)
        # get neighbouring cell in grid which is set as goal
        goal_u, goal_v = get_closest_free_cell(agent_u, agent_v, gridmax_u, gridmax_v, grid, max_item_goal_dist=0.6,
                                               required_item_goal_dist=0.2)
        goal_x, goal_y = grid.get_position_by_indices(goal_u, goal_v)
        final_theta_ref = auxiliary_functions.angle_consistency(math.atan2(bmax_y - goal_y, bmax_x - goal_x))
        goal_ref = [goal_x, goal_y, final_theta_ref]
        return goal_ref

    def is_goalref_still_valid(self, s, agent_x, agent_y, agent_theta, grid, b0=-1):
        # get cell in belief grid with highest belief
        bmax_x, bmax_y = b0.get_max_belief_cell_pos(node_nr=b0.get_node_nr(agent_x, agent_y), item_nr=self.item_i)
        # close enough to target cell already?
        dist = math.sqrt((agent_x - bmax_x) ** 2 + (agent_y - bmax_y) ** 2)
        theta_ref = auxiliary_functions.angle_consistency(math.atan2(bmax_y - agent_y, bmax_x - agent_x))
        delta_theta = math.fabs(auxiliary_functions.angle_consistency(theta_ref - agent_theta))
        if dist < 1.0 and delta_theta < 0.2:
            return False
        # is reference goal and target cell still the same?
        gridmax_u, gridmax_v = grid.get_cell_indices_by_position(bmax_x, bmax_y)
        goal_x, goal_y = self.goal_ref[0], self.goal_ref[1]
        goal_u, goal_v = grid.get_cell_indices_by_position(goal_x, goal_y)
        return math.fabs(goal_u - gridmax_u) <= 1 and math.fabs(goal_v - gridmax_v) <= 1

    def is_action_finished(self, s, item_map, b0):
        xa = s['xa']
        item_type = auxiliary_functions.inverse_item_map(self.item_i, item_map)
        if s[item_type] == 'agent':
            return True, True
        # measure belief drop
        if config.simulation_name == 'low_fidelity_simulation':
            b_tot = b0.get_aggregated_belief(node_nr=xa)[self.item_i]
            if b_tot / self.b0_tot[self.item_i] < 0.25:
                return True, False
        # check if belief of another item increased by a lot
        if config.simulation_name == 'low_fidelity_simulation':
            b_max = 0
            for idx in range(len(s) - 1):
                if idx == self.item_i:
                    continue
                b_max_idx = b0.get_max_belief_cell_value(xa, idx)
                if b_max_idx > b_max:
                    b_max = b_max_idx
            if (b_max / self.start_max_belief) > (max(1.05, 0.9 / self.start_max_belief)):
                self.log.info('pickup finihsed, max belief over 0.9')
                return True, True
            else:
                return False, False
        else:
            return False, False


class Release(Action):
    def __init__(self, name, file_name, layer, environment, xa_values, nodegraph):
        Action.__init__(self, name, file_name, layer, environment, xa_values, nodegraph)
        self.log = logging.getLogger(__name__)

    def initialize_t_expected(self, xa_values, nodegraph):
        # read from file
        with open(self.file_name_t_expected) as f:
            time_list = f.readline().strip()
            self.t_expected = [float(val) for val in time_list.split(' ')]
        return

    def core_algorithm(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        # sanity check: is agent carrying anything?
        if 'agent' not in s.values():
            return 'error: requirements for subroutine release are not met'
        agent_u, agent_v = grid.get_cell_indices_by_position(agent_x, agent_y)
        item_i = 0
        for key in s.keys():
            if s[key] == 'agent':
                item_i = key
                break
        item = items[item_map[item_i]]
        # if not in goal_node, just release the item
        if nodegraph.get_node_nr(item.goal_x, item.goal_y) != s['xa']:
            goal_ref = 'release_item'
            return goal_ref
        g_u, g_v = grid.get_cell_indices_by_position(item.goal_x, item.goal_y)
        u_ref, v_ref = get_closest_free_cell(agent_u, agent_v, g_u, g_v, grid, max_item_goal_dist=0.6,
                                             required_item_goal_dist=0.2)
        x_ref, y_ref = grid.get_position_by_indices(u_ref, v_ref)
        theta_ref = auxiliary_functions.angle_consistency(math.atan2(item.goal_y - y_ref, item.goal_x - x_ref))

        dist = math.sqrt((item.goal_x - agent_x) ** 2 + (item.goal_y - agent_y) ** 2)
        dist_theta = auxiliary_functions.angle_consistency(theta_ref - agent_theta)
        if dist < config.robot_range - 0.1 and dist_theta < 0.2:
            goal_ref = 'release_item'
            return goal_ref
        goal_ref = [x_ref, y_ref, theta_ref]
        return goal_ref

    def is_goalref_still_valid(self, s, agent_x, agent_y, agent_theta, grid, b0=-1):
        if self.goal_ref != 'release_item':
            x_ref, y_ref, theta_ref = self.goal_ref
            dist = math.sqrt((x_ref - agent_x) ** 2 + (y_ref - agent_y) ** 2)
            dist_theta = auxiliary_functions.angle_consistency(theta_ref - agent_theta)
            if dist < 0.4 and dist_theta < 0.2:
                return False
        else:
            return True

    def is_action_finished(self, s, item_map, b0=-1):
        action_finished = True
        for item_type in s.keys():
            if item_type == 'xa':
                continue
            if s[item_type] == 'agent':
                action_finished = False

        if action_finished:
            return True, False
        else:
            return False, False


class LookAround(Action):
    def __init__(self, name, file_name, layer, environment, xa_values, nodegraph, node_mapping, grid_cell_width,
                 grid_cell_height):
        self.node_mapping = node_mapping
        self.current_max_belief = 0
        self.initial_median_belief = 0
        self.initial_avr_max_belief = 0
        self.steps_since_replanning = 0
        # keep list of grids, one grid per node/xa_value
        self.layer = layer
        self.node_grids = []
        self.initialize_grids(xa_values, nodegraph, grid_cell_width, grid_cell_height, environment)
        Action.__init__(self, name, file_name, layer, environment, xa_values, nodegraph)
        # read from file
        with open(self.file_name_obs_prob) as f:
            obs_prob_values = f.readline().strip()
            value_list = [float(val) for val in obs_prob_values.split(' ')]
            for idx, value in enumerate(value_list):
                self.observation_probabilities[idx] = value
        self.log = logging.getLogger(__name__)

    def initialize_grids(self, xa_values, nodegraph, cell_width, cell_height, environment):
        for xa in xa_values:
            # create a local grid
            min_x0 = min([rec.x0 for rec in nodegraph.node_recs[xa]])
            max_x1 = max([rec.x0 + rec.width for rec in nodegraph.node_recs[xa]])
            min_y0 = min([rec.y0 for rec in nodegraph.node_recs[xa]])
            max_y1 = max([rec.y0 + rec.height for rec in nodegraph.node_recs[xa]])

            nr_of_cells_x = int((max_x1 - min_x0) / cell_width)
            nr_of_cells_y = int((max_y1 - min_y0) / cell_height)
            grid = Grid(nr_of_cells_x, nr_of_cells_y, world_width=max_x1 - min_x0, world_height=max_y1 - min_y0,
                        x0=min_x0, y0=min_y0, default_value='occupied')
            # set 'free' regions
            for rec_idx, rec in enumerate(nodegraph.node_recs[xa]):
                # grid.set_region(rec, 'free')
                if environment == 'small':
                    if self.layer == 1:
                        if xa == 2:
                            grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='smaller', y0_beh='smaller',
                                            y1_beh='bigger')
                        elif xa == 5:
                            grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='bigger', y0_beh='smaller',
                                            y1_beh='smaller')
                        else:
                            grid.set_region(rec, 'free', x0_beh='smaller', x1_beh='smaller', y0_beh='smaller',
                                            y1_beh='smaller')
                elif environment == 'big':
                    if self.layer == 2:
                        if rec_idx == 0 and (xa == 1 or xa == 4 or xa == 7 or xa == 10):
                            grid.set_region(rec, 'free', y1_beh='bigger')
                        elif xa == 13 or xa == 16 or xa == 19 or xa == 22:
                            grid.set_region(rec, 'free', y1_beh='bigger')
                        else:
                            grid.set_region(rec, 'free')
            self.node_grids.append(grid)

    def initial_call_function(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        self.start_xa = s['xa']
        # get start-belief of this node
        self.b0_tot = b0.get_aggregated_belief(self.start_xa)
        xa_u, xa_v = self.node_grids[self.start_xa].get_cell_indices_by_position(x=agent_x, y=agent_y)
        self.node_grids[self.start_xa].cells[xa_v][xa_u].seen = 0
        self.start_max_belief = b0.get_max_belief_cell_value(self.start_xa)
        self.current_max_belief = self.start_max_belief
        self.initial_median_belief = b0.get_median_belief(node_nr=self.start_xa)
        self.initial_avr_max_belief = np.sum(b0.get_N_max_belief_values(node_nr=self.start_xa, N=100)) / 100
        self.goal_ref = self.core_algorithm(s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0)
        self.initial_call = False

    def initialize_t_expected(self, xa_values, nodegraph):
        # read from file
        with open(self.file_name_t_expected) as f:
            times = [line.strip() for line in f]
        for time_list in times:
            self.t_expected.append([float(val) for val in time_list.split(' ')])
        return

    def finish_action(self, s, item_present):
        self.log.info('Action is finished')
        Action.finish_action(self, s, item_present)
        for grid in self.node_grids:
            grid.set_all_seen_values(value=1)

    def is_action_finished(self, s, item_map, b0):
        xa = s['xa']
        # criteria 1: if the max_belief increased, finish
        if config.simulation_name == 'low_fidelity_simulation':
            if self.start_max_belief > 0:
                b_max = b0.get_max_belief_cell_value(xa)
                if (b_max / self.start_max_belief) > (max(1.05, 0.9 / self.start_max_belief)):
                    self.log.info(self.name + ' finished, max belief over 0.9')
                    return True, True
        # criteria 2: if at least 90 % of the cells were observed: finish
        nr_of_obs_cells = 0
        for v in range(len(self.node_grids[xa].cells)):
            for u in range(len(self.node_grids[xa].cells[0])):
                if self.node_grids[xa].cells[v][u].seen < 1:
                    nr_of_obs_cells += 1
        total_nr_of_cells = self.node_grids[xa].get_nr_of_free_cells()
        if nr_of_obs_cells / total_nr_of_cells > 0.9:
            self.log.info(self.name + ' finished, 90% of cells within this node are observed')
            return True, False
        # critera 3: if overall belief changed strongly enough
        if config.simulation_name == 'low_fidelity_simulation':
            if self.start_max_belief > 0:
                # get median belief
                median_belief = b0.get_median_belief(node_nr=xa)
                # get average of the highest belief cells
                max_belief_values = b0.get_N_max_belief_values(node_nr=xa, N=100)
                avr_max_belief = np.sum(max_belief_values) / len(max_belief_values)
                if avr_max_belief - median_belief < (1 / 20.0) * (
                        self.initial_avr_max_belief - self.initial_median_belief):
                    self.log.info(self.name + ' finished, peak reduced by {}'.format(
                        (self.initial_avr_max_belief - self.initial_median_belief) / avr_max_belief - median_belief))
                    return True, False
        return False, False

    def is_goalref_still_valid(self, s, agent_x, agent_y, agent_theta, grid, b0=-1):
        # check if belief changed strongly enough
        b_max = b0.get_max_belief_cell_value(s['xa'])
        if b_max > 1.1 * self.current_max_belief:
            return False
        # check if not close enough to target reference
        else:
            x_ref, y_ref, theta_ref = self.goal_ref
            dist = math.sqrt((x_ref - agent_x) ** 2 + (y_ref - agent_y) ** 2)
            dist_theta = auxiliary_functions.angle_consistency(theta_ref - agent_theta)
            if dist < 0.4 and dist_theta < 0.2:
                return False
            else:
                return True

    def update_local_grid(self, xa, o_cells):
        for o_c in o_cells:
            u, v = self.node_grids[xa].get_cell_indices_by_position(o_c.x, o_c.y)
            if u != -1:
                self.node_grids[xa].cells[v][u].seen = 0

    # careful: if something goes wrong this function may return a empty list
    def core_algorithm(self, s, item_map, agent_x, agent_y, agent_theta, grid, nodegraph, items, b0=-1):
        xa = s['xa']
        goal_ref = []
        agent_local_u, agent_local_v = self.node_grids[xa].get_cell_indices_by_position(agent_x, agent_y)
        # check if max belief increased since last time
        b_max = b0.get_max_belief_cell_value(xa)
        if b_max > 1.1 * self.current_max_belief:
            self.current_max_belief = b_max
            goal_ref = [agent_x, agent_y, agent_theta]
            return goal_ref
        # STEP 1: find frontier cells (=candidate cells)
        candidate_u, candidate_v, candidate_theta = self.find_frontier_cells(xa)
        # STEP 2: loop over all candidate cells and compute their delta_t, sum_b values
        sum_b_arr = []
        for idx in range(len(candidate_u)):
            cell_u, cell_v, cell_theta = candidate_u[idx], candidate_v[idx], candidate_theta[idx]
            # STEP 2a) compute expected number of new observed cells
            sum_b_arr += [self.get_sum_b(xa, cell_u, cell_v, cell_theta, grid=self.node_grids[xa], b0=b0)]
            # STEP 2b) compute time needed to travel to candidate cell
        delta_t_arr = self.get_delta_t_euclidian_numpy(agent_local_u, agent_local_v, agent_theta, candidate_u,
                                                       candidate_v, candidate_theta, grid=self.node_grids[xa])
        # STEP 3: calculate value function for all candidate cells
        u_sum_b = (np.array(sum_b_arr) - 0.0000000001) / (max(sum_b_arr) - 0.0000000001)
        u_delta_t = np.ones(delta_t_arr.shape) - (delta_t_arr - min(delta_t_arr)) / (
                max(delta_t_arr) - min(delta_t_arr))
        candidate_value = 0.5 * u_sum_b + 0.5 * u_delta_t
        idx_best = np.argmax(candidate_value)
        candidate_x, candidate_y = self.node_grids[xa].get_position_by_indices(candidate_u[idx_best],
                                                                               candidate_v[idx_best])
        goal_ref = [candidate_x, candidate_y, auxiliary_functions.angle_consistency(candidate_theta[idx_best])]
        return goal_ref

    def find_frontier_cells(self, xa):
        u_arr, v_arr, theta_arr = [], [], []
        for v in range(0, len(self.node_grids[xa].cells)):
            for u in range(0, len(self.node_grids[xa].cells[0])):
                # if cell is a potential frontier cell
                if self.node_grids[xa].get_cell_value_by_index(u=u, v=v) == 'free' and \
                        self.node_grids[xa].cells[v][u].seen == 1:
                    # check 8-neighbourhood to see if the cell is neighbouring an already observed cell
                    leave_neighbourhood_loop = False
                    for dv in range(-1, 2):
                        for du in range(-1, 2):
                            if 0 <= u + du < self.node_grids[xa].nr_of_cells_x and 0 <= v + dv < self.node_grids[
                                xa].nr_of_cells_y and \
                                    self.node_grids[xa].cells[v + dv][u + du].seen < 1:
                                # for every frontier cell there are 8 possible angles
                                theta_values = np.arange(0, 2 * math.pi - 0.001, math.pi / 4)
                                for theta in theta_values:
                                    u_arr += [u]
                                    v_arr += [v]
                                    theta_arr += [theta]
                                # frontier_cells[[u, v]] = 0
                                leave_neighbourhood_loop = True
                                break
                        if leave_neighbourhood_loop:
                            break
        return np.array(u_arr), np.array(v_arr), np.array(theta_arr)

    def get_delta_t_euclidian_numpy(self, agent_u, agent_v, agent_theta, cell_u_arr, cell_v_arr, cell_theta_arr, grid):
        theta_ij_arr = 2 * math.pi - np.arctan2(cell_v_arr - agent_v, cell_u_arr - agent_u)
        delta_theta = np.abs(agent_theta - theta_ij_arr) % (2 * math.pi)
        delta_theta = np.minimum(delta_theta, 2 * math.pi - delta_theta)
        time_rot1 = delta_theta / config.robot_angular_speed
        # time for driving towards candidate cell
        dist = np.sqrt(
            (grid.cell_width * (agent_u - cell_u_arr)) ** 2 + (grid.cell_height * (agent_v - cell_v_arr)) ** 2)
        time_dist = dist / config.robot_speed
        # time for rotating in candidate-cell-configuration
        delta_theta2 = np.abs(cell_theta_arr - theta_ij_arr) % (2 * math.pi)
        delta_theta2 = np.minimum(delta_theta2, 2 * math.pi - delta_theta2)
        time_rot2 = delta_theta2 / config.robot_angular_speed

        return time_rot1 + time_dist + time_rot2

    def get_sum_b(self, xa, cell_u, cell_v, cell_theta, grid, b0):
        # estimate/simulate measurements
        # get transformation function of robot:
        x, y = grid.get_position_by_indices(cell_u, cell_v)
        trans_matrix = np.array([[math.cos(cell_theta), math.sin(cell_theta), x],
                                 [-math.sin(cell_theta), math.cos(cell_theta), y],
                                 [0, 0, 1]])
        # get transformed viewing cone points
        observations = []
        line_resolution = 0.195
        for line in config.robot_viewing_cone[0::3]:
            # get homogenous points of first and last one of line
            p_0h = np.array([line[0][0], line[0][1], 1])
            p_nh = np.array([line[1][0], line[1][1], 1])
            # get transformed points
            p_0trans = np.dot(trans_matrix, p_0h)
            p_ntrans = np.dot(trans_matrix, p_nh)
            # get unit vector pointing in line direction
            vec = np.array([p_ntrans[0] - p_0trans[0], p_ntrans[1] - p_0trans[1]])
            vec = 1. / (LA.norm(vec)) * vec
            for i in range(0, int(math.ceil(3 / line_resolution) + 1)):
                p_i = p_0trans[0:2] + i * line_resolution * vec
                u_grid, v_grid = grid.get_cell_indices_by_position(p_i[0], p_i[1])
                if u_grid == -1 or v_grid == -1:
                    break
                # check if observation is inside node that we want to observe
                if b0.is_point_in_node(x=p_i[0], y=p_i[1], node_nr=xa) and grid.cells[v_grid][u_grid].seen == 1:
                    observations += [(p_i[0], p_i[1])]
        sum_b = sum(b0.belief.get_total_belief_of_observations(observations))
        return sum_b
