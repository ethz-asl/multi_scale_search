from __future__ import division # required for standard float division in python 2.7
import config
from src.auxiliary_files.grid import Grid
import numpy as np
import copy
import os
import time
import xml.etree.ElementTree as ET
from src.auxiliary_files.grid import Grid
from src.multi_scale_search import auxiliary_functions
from src.multi_scale_search.actions import LookAround
from src.multi_scale_search.actions import Navigate
from src.multi_scale_search.actions import Pickup
from src.multi_scale_search.actions import Release
from src.multi_scale_search.belief import Belief
from src.multi_scale_search.belief import BeliefRepresentation


# core.py consists of:
# - Cell-observation class
# - POMDP_Problem datastructure class
# - Nodegraph class
# - Agent base class
# - MultiScaleAgent base class

class O_Cell(object):
    def __init__(self, x, y, obs_val):
        self.x = x
        self.y = y
        self.obs_val = obs_val


# this class is a datastructure to hold all state, observation, action variables, terminal states and the solution alpha vectors of a POMDP problem
class POMDPProblem(object):
    def __init__(self, xa_values_li, x_items_values_li, s_terminals_li, z_values_li, action_names_li):
        self.xa_values = copy.copy(xa_values_li)
        self.x_items_values = copy.copy(x_items_values_li)
        self.s_terminals = copy.copy(s_terminals_li)
        self.z_values = copy.copy(z_values_li)
        self.action_names = copy.copy(action_names_li)
        # initialize alpha vectors
        self.alpha_vectors = np.array([], dtype=float)
        self.alpha_vectors_attrib = np.array([], dtype=int)

    def set_variable_sets(self, xa_values_li, x_items_values_li, s_terminals_li, z_values_li, action_names_li):
        self.xa_values = copy.copy(xa_values_li)
        self.x_items_values = copy.copy(x_items_values_li)
        self.s_terminals = copy.copy(s_terminals_li)
        self.z_values = copy.copy(z_values_li)
        self.action_names = copy.copy(action_names_li)

    def set_alpha_vectors(self, alpha_vectors, alpha_vectors_attrib):
        self.alpha_vectors = copy.copy(alpha_vectors)
        self.alpha_vectors_attrib = copy.copy(alpha_vectors_attrib)


class NodeGraph(object):
    # node_recs is a dict with key = node_nr, value  = list of rectangles
    def __init__(self, nr_of_nodes=9, node_recs={}):
        self.node_recs = node_recs
        # graph is dictionary with key = node-number and value = list of node-numbers the node is connected to
        self.graph = {}
        for n in range(nr_of_nodes):
            self.graph[n] = []

    def get_node_nr(self, x, y):
        for item in self.node_recs.items():
            for rec in item[1]:
                if rec.is_point_in_rec(x, y):
                    return item[0]
        print('invalid x,y = ({}, {}) input in NodeGraph.get_node_nr'.format(x, y))
        return -1

    def get_all_recs_as_flat_list(self):
        recs = []
        for node_nr in self.node_recs:
            recs += self.node_recs[node_nr]
        return recs

    def get_center_of_gravity(self, node_nr):
        x_mid_points, y_mid_points, weights = [], [], []
        for rec in self.node_recs[node_nr]:
            x,y = rec.get_mid_point()
            x_mid_points.append(x)
            y_mid_points.append(y)
            weights.append(rec.get_area())
        x_cog = np.dot(x_mid_points, weights) / sum(weights)
        y_cog = np.dot(y_mid_points, weights) / sum(weights)
        return x_cog, y_cog

    # indexing for user starts at 0
    def add_edge(self, n_i, n_j):
        self.graph[n_i].append(n_j)
        self.graph[n_j].append(n_i)

    def get_nav_actions(self, n):
        nav_actions = []    # list of strings
        connected_nodes = self.graph[n]
        for i in connected_nodes:
            if i < n:
                nav_actions.append('nav(' + str(i) + ',' + str(n) + ')')
            else:
                nav_actions.append('nav(' + str(n) + ',' + str(i) + ')')
        return nav_actions

    # nodes_restriction is a list of nodes, the navigation action has to go to a node in node_restriction
    def get_nav_actions_restricted(self, n, nodes_restriction):
        nav_actions = []
        connected_nodes = self.graph[n]
        for i in connected_nodes:
            if i not in nodes_restriction:
                continue
            elif i < n:
                nav_actions.append('nav(' + str(i) + ',' + str(n) + ')')
            else:
                nav_actions.append('nav(' + str(n) + ',' + str(i) + ')')
        return nav_actions

    def get_all_nav_actions(self, action_names):
        for n in self.graph.keys():
            nav_n_actions = self.get_nav_actions(n)
            # add nav_n_actions to nav_actions but delete duplicates
            for a in nav_n_actions:
                if a not in action_names:
                    action_names.append(a)

    # nodes = list of node_nrs for which the navigation actions connecting the nodes is returned
    # nodes_restriction is a list/set of nodes to which the navigation function are allowed
    def get_nav_actions_for_nodes(self, nodes, nodes_restriction):
        nav_actions = []
        for n in nodes:
            nav_n_actions = self.get_nav_actions_restricted(n, nodes_restriction)
            for a in nav_n_actions:
                if a not in nav_actions:
                    nav_actions.append(a)
        return nav_actions

    def get_neighbour_nodes(self, node_nr):
        return self.graph[node_nr]


# this is a base-class for agents.
# The agents differentiate though:
#   - method of interpreting the observations
#   - algorithm that chooses the actions given the observations
class Agent(object):

    # conf0 is a tuple of x-position (world-coordinates), y-position, theta (robot-orientation)
    def __init__(self, grid=Grid(1, 1, 1, 1), pose0=(0, 0, 0), environment='Null'):
        self.grid = grid
        self.x = pose0[0]
        self.y = pose0[1]
        self.theta = pose0[2]
        self.u, self.v = self.grid.get_cell_indices_by_position(self.x, self.y)
        self.grid.cells[self.v][self.u].seen = 0    # this cell is observed
        self.carries = 'none'    # at start agent does not carry anything
        self.environment = environment
        # initialize the list of items the agent is looking for and the goal (right now its only 'table')
        self.task = {}
        self.task_finished = []
        self.nr_of_items = 1
        self.items = []     # list of items. its up to the specific agent how to represent items and furniture
        self.replanning = False  # this is only needed for the documentation in the end
        self.computation_time = 0
        self.computation_time_for_each_action = []
        self.k = 0  # discrete timestep
        self.timeout_time = 10.0
        self.item_map = {}
        self.initialized = False
        # for POMDP agents:
        self.subroutine_actions_history = {}
        self.belief = None

    def set_pose(self, x, y, theta, carry='none'):
        print('x={}, y={}, theta={}, carry={}'.format(x, y, theta, carry))
        self.x = x
        self.y = y
        self.theta = theta
        self.carries = carry

    def set_timeout_time(self, timeout_time):
        self.timeout_time = timeout_time

    def start_solving(self):
        self.initialized = True

    # give the agent a task. task = dictionary with keys = items and value = goal
    def set_task(self, task):
        self.task = task.copy()

    # gets overwritten from child classes, since agents have different item representation
    def set_item(self, item_type, item_x, item_y):
        do_nothing = True

    def add_item(self, item_type, goal_x, goal_y):
        do_nothing = True

    # gets overwritten from child classes
    def set_belief(self, x_list, y_list, b_list, b_sigma, item_nr_list):
        do_nothing = True

    # gets overwritten from child classes
    def update_carry(self, o_carry, item_x=None, item_y=None):
        do_nothing = True

    # gets overwritten from child classes
    def update_pose(self, o_x, o_y, o_theta):
        do_nothing = True

    # gets overwritten from child classes
    def interpret_observations(self, o_cells):
        do_nothing = True

    # returns goal_ref, gets overwritten from child classes
    def choose_action(self):
        goal_ref = [0.5, 0.5, 0.0]
        return goal_ref

    def set_replanning(self, replanning):
        self.replanning = replanning

    # gets overwritten by POMDP agents
    def set_grid_state_data(self, colormap):
        N_x, N_y = self.grid.nr_of_cells_x, self.grid.nr_of_cells_y
        data = np.ones((N_x, N_y))
        for j in range(0, N_y):
            for i in range(0, N_x):
                data[i, j] = self.grid.cells[j][i].seen
        # make colormap
        colormap.set_data(data)


class Item:
    def __init__(self, name, x, y, goal_x=-1.0, goal_y=-1.0, goal_nodes_layers=None):
        self.name = name
        self.x = x
        self.y = y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_nodes_layers = goal_nodes_layers

    def set_goal_xy(self, goal_x, goal_y, goal_nodes_layers):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_nodes_layers = goal_nodes_layers


# this is a base-class for MultiScale Agents
class AgentMultiScaleBasis(Agent):
    def __init__(self, grid=Grid(1, 1, 1, 1), conf0=(0, 0, 0), nr_of_layers=2, node_mapping=None, rec_env=None,
                 item_names=('mug'), environment='None'):
        Agent.__init__(self, grid, conf0, environment)
        self.timeout_time = 1.0
        self.name = 'MultiScaleBasis'
        self.b0_threshold = 0.999
        self.nr_of_items = len(item_names)
        self.nr_of_layers = nr_of_layers
        self.POMDPs = []
        self.node_mapping = node_mapping
        self.file_names_input = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM_input_l{}.pomdpx'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.file_names_output = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM_output_l{}.policy'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        # create rectangles of nodes and nodegraph
        self.nodegraphs_layers = []
        for layer in range(nr_of_layers):
            if layer == nr_of_layers - 1:
                node_recs_layer = auxiliary_functions.get_recs_of_nodes(rec_env, environment=self.environment,
                                                                        nr_of_layers=self.nr_of_layers)
                self.nodegraphs_layers.append(NodeGraph(
                    nr_of_nodes=sum([len(node_mapping[layer - 1][key]) for key in node_mapping[layer - 1].keys()]),
                    node_recs=node_recs_layer))
                # config.draw_rectangles(self.nodegraphs_layers[layer].get_all_recs_as_flat_list())
            else:
                self.nodegraphs_layers.append(NodeGraph(nr_of_nodes=len(node_mapping[layer].keys())))
            auxiliary_functions.construct_nodegraph_multiscale(nodegraph=self.nodegraphs_layers[-1], layer=layer,
                                                               environment=self.environment)
        self.nr_of_nodes = [len(self.nodegraphs_layers[layer].graph.keys()) for layer in range(self.nr_of_layers)]
        self.current_action_name_lN = 'none'  # this is the actual subroutine
        # create item-mapping
        self.item_map = {}
        for item_idx, item_name in enumerate(item_names):
            self.item_map[item_name] = item_idx
            self.items.append(Item(name=item_name, x=-1, y=-1))

        # create list of dicts of actions, each list entry corresponds to action_dict of a layer,
        # key = action_name, value = action object
        self.subroutine_actions = {}  # this are the actual subroutine actions
        self.action_names_layers = []
        self.xa_values_all_layers = []
        self.xi_values_all_layers = []
        self.zi_values_all_layers = []
        for layer in range(self.nr_of_layers):
            self.xa_values_all_layers.append(list(self.nodegraphs_layers[layer].graph.keys()))
            self.xi_values_all_layers.append(list(self.nodegraphs_layers[layer].graph.keys()) + ['agent', 'goal'])
            self.zi_values_all_layers.append(['no'] + list(self.nodegraphs_layers[layer].graph.keys()) + ['agent'])
            action_names_layer = []
            self.nodegraphs_layers[layer].get_all_nav_actions(action_names_layer)
            if layer == self.nr_of_layers - 1:
                action_names_layer.append('look_around')
            for item_name in item_names:
                action_names_layer.append('pickup{}'.format(self.item_map[item_name]))
            action_names_layer.append('release')
            self.action_names_layers.append(action_names_layer)
        for action_name in self.action_names_layers[-1]:
            if action_name[0:3] == 'nav':
                self.subroutine_actions[action_name] = Navigate(name=action_name,
                                                                file_name=action_name + '.txt',
                                                                layer=self.nr_of_layers - 1,
                                                                environment=self.environment,
                                                                xa_values=self.xa_values_all_layers[-1],
                                                                nodegraph=self.nodegraphs_layers[-1], grid=self.grid)
            elif action_name[0:6] == 'pickup':
                self.subroutine_actions[action_name] = Pickup(name=action_name,
                                                              file_name='pickup.txt',
                                                              layer=self.nr_of_layers - 1, environment=self.environment,
                                                              file_name_look_around='look_around.txt',
                                                              xa_values=self.xa_values_all_layers[-1],
                                                              nodegraph=self.nodegraphs_layers[-1])
            elif action_name == 'look_around':
                self.subroutine_actions[action_name] = LookAround(name=action_name,
                                                                  file_name='look_around.txt',
                                                                  layer=self.nr_of_layers - 1,
                                                                  environment=self.environment,
                                                                  xa_values=self.xa_values_all_layers[-1],
                                                                  nodegraph=self.nodegraphs_layers[-1],
                                                                  node_mapping=self.node_mapping,
                                                                  grid_cell_width=self.grid.cell_width,
                                                                  grid_cell_height=self.grid.cell_height)
            elif action_name == 'release':
                self.subroutine_actions[action_name] = Release(name=action_name,
                                                               file_name='release.txt',
                                                               layer=self.nr_of_layers - 1,
                                                               environment=self.environment,
                                                               xa_values=self.xa_values_all_layers[-1],
                                                               nodegraph=self.nodegraphs_layers[-1])
        self.subroutine_actions_history = [{} for l in range(self.nr_of_layers)]  # key = timestep k, value = action

        # initialize belief grid
        self.belief = Belief(total_nr_of_cells=10000 * (grid.total_width * grid.total_height) /
                                               (config.grid_ref_width * config.grid_ref_height),
                             recs_beliefgrids=self.nodegraphs_layers[-1].get_all_recs_as_flat_list(),
                             nr_of_items=len(item_names))
        # initialize b0_layer in a list
        self.b0_layers = []
        # config.draw_rectangles(self.nodegraphs_layers[-1].get_all_recs_as_flat_list())
        for layer in range(self.nr_of_layers):
            nodes_recs_mapping = auxiliary_functions.get_nodes_recs_mapping(layer, self.nr_of_layers, self.node_mapping,
                                                                            self.nodegraphs_layers[-1])
            self.b0_layers.append(BeliefRepresentation(belief=self.belief, nodes_recs_mapping=nodes_recs_mapping))
            # for key in nodes_recs_mapping.keys():
            #     rec_list = []
            #     for idx in nodes_recs_mapping[key]:
            #         rec_list.append(self.belief.belief_grids[idx].rec)
            #     config.draw_rectangles(rec_env + rec_list)
        # initialize state_list, where each entry is the state of a layer, a state is a dict: {'xa': val, 0: val, ...}
        self.s_layers = []
        for layer in range(self.nr_of_layers):
            self.s_layers.append({'xa': self.pos_to_node(conf0[0], conf0[1], layer)})
            for item_name in item_names:
                self.s_layers[-1][item_name] = -1
        # read open-loop policies for estimating reward and observation model of all layers (except layer N)
        self.open_loop_policies = self.read_open_loop_policies()
        # initialize expected times and observation probabilities for all layers
        self.timerewards_layers = []  # list of dictionaries: {(s_li, a_li):t:expected}
        self.observation_probabilities_layers = []  # list of dictionaries: {(z_li, s_li, a_li):t:expected}
        # variable with values 'no', 'current', 'next' that indicates ib POMDP above chooses a nav. action with multiple terminal states in the layer below
        self.solve_terminal_states_problems = 'no'
        # computation time tracking
        self.computation_time_layers = []
        for layer in range(nr_of_layers):
            self.computation_time_layers.append(0)

    def start_solving(self):
        self.initialized = True
        for key in self.item_map:
            print('item {}: {}'.format(key, self.item_map[key]))
            item = self.items[self.item_map[key]]
            print('item_goal = {}'.format(item.goal_nodes_layers[-1]))
            print('belief:')
            for n in range(self.nr_of_nodes[-1]):
                b_vec = self.belief.get_aggregated_belief(rec_nr=n)
                print('b(n={})={}'.format(n, b_vec))
            self.compute_timerewards_for_all_layers()
            self.compute_observation_probabilities_for_all_layers()

    # currently not used, might be useful for future implementation (e.g. ros)
    def add_item(self, item_type, goal_x=-1.0, goal_y=-1.0):
        if item_type not in self.item_map:
            for layer in range(self.nr_of_layers):
                self.s_layers[layer][item_type] = -1
            self.item_map[item_type] = len(self.items)
            goal_nodes_layers = [self.pos_to_node(goal_x, goal_y, layer) for layer in range(0, self.nr_of_layers)]
            self.items.append(
                Item(name=item_type, x=-1, y=-1, goal_x=goal_x, goal_y=goal_y, goal_nodes_layers=goal_nodes_layers))
            self.nr_of_items = len(self.items)
            self.belief.add_item()
            for layer in range(self.nr_of_layers):
                self.b0_layers[layer].set_belief(self.belief)
            # add pickup action for the item
            action_name = 'pickup{}'.format(self.item_map[item_type])
            self.subroutine_actions[action_name] = Pickup(
                name=action_name, file_name='pickup.txt', layer=self.nr_of_layers - 1,
                file_name_look_around='look_around.txt', environment=self.environment,
                xa_values=self.xa_values_all_layers[-1], nodegraph=self.nodegraphs_layers[-1])
            for layer in range(self.nr_of_layers):
                self.action_names_layers[layer].append(action_name)

    def set_belief(self, x_list, y_list, b_list, b_sigma, item_nr_list):
        for i in range(len(x_list)):
            self.belief.add_belief_spot(x_list[i], y_list[i], b_list[i], b_sigma, item_nr_list[i])

    # this function is not used atm. However, it could be handy for future implementations
    def add_belief_spot(self, x, y, b, sigma, item_type):
        if item_type not in self.item_map:
            print('you first need to add the item to the world before setting the belief')
            return 'warning'
        item_nr = self.item_map[item_type]
        self.belief.add_belief_spot(mu_x=x, mu_y=y, prob=b, sigma=sigma, item_nr=item_nr)
        return 'success'

    def inverse_item_map(self, item_nr):
        for item_type in self.item_map.keys():
            if self.item_map[item_type] == item_nr:
                return item_type
        return 'none'

    def pos_to_node(self, x, y, layer, verbose=False):
        if verbose:
            print(self.b0_layers[layer])
        node_nr = self.b0_layers[layer].get_node_nr(x, y)
        if verbose:
            print(node_nr)
        if node_nr == -1:
            print('error in pos_to_node mapping')
            return -1
        else:
            return node_nr

    def get_subnodes(self, node, layer):
        subnodes = copy.copy(self.node_mapping[layer][node])
        subnodes.sort()
        return subnodes

    def get_subnodes_over_multiple_layers(self, top_node, top_layer, bottom_layer):
        subnodes_bottom_layer = copy.copy(self.node_mapping[top_layer][top_node])
        for layer in range(top_layer + 1, bottom_layer):
            subnodes_layer = []
            for subnode in subnodes_bottom_layer:
                subnodes_layer += copy.copy(self.node_mapping[layer][subnode])
            subnodes_bottom_layer = copy.copy(subnodes_layer)
        subnodes_bottom_layer.sort()
        return subnodes_bottom_layer

    # return the var from one layer above layer_lower to which var_lower belongs to
    def var_lower_to_var_top(self, layer_lower, var_lower):
        if var_lower == 'agent' or var_lower == 'goal' or var_lower == 'not_here' or var_lower == '*':
            return var_lower
        else:
            for node_top, nodes_lower in self.node_mapping[layer_lower - 1].items():
                if var_lower in nodes_lower:
                    return node_top
        print('invalid layer_lower={}, var_lower={} input to var_lower_to_var_top()'.format(layer_lower, var_lower))

    def var_lower_to_var_top_multiple_layers(self, layer_lower, var_lower, top_layer):
        if var_lower == 'agent' or var_lower == 'goal' or var_lower == 'not_here' or var_lower == '*':
            return var_lower
        else:
            node = var_lower
            for layer in range(layer_lower, top_layer, -1):
                for node_top, nodes_lower in self.node_mapping[layer - 1].items():
                    if node in nodes_lower:
                        node = node_top
            return node

    def get_subobservations(self, z_li, layer):
        if z_li == 'no':
            return ['no']
        elif z_li == 'agent':
            return ['agent']
        else:
            return self.get_subnodes(node=z_li, layer=layer)

    def set_b0_threshold(self, b0_threshold):
        self.b0_threshold = b0_threshold

    def set_task(self, task):
        self.task = task.copy()
        # check if item_map and items list is coherent
        is_coherent = True
        if len(self.task.keys()) != len(self.items):
            is_coherent = False
        for item_type in self.task.keys():
            if item_type not in self.item_map:
                is_coherent = False
            # set goal of item
            g_x, g_y = self.task[item_type]
            goal_nodes_layers = [self.pos_to_node(g_x, g_y, layer, verbose=True) for layer in
                                 range(0, self.nr_of_layers)]
            self.items[self.item_map[item_type]].set_goal_xy(g_x, g_y, goal_nodes_layers)
        if not is_coherent:
            print('something went wrong when setting items')
            return 'error'
        else:
            return 'success'

    def compute_timerewards_for_all_layers(self):
        # start with the second lowest layer and compute t_expected with lowest layer models.
        for layer in range(self.nr_of_layers - 2, -1, -1):
            timereward_layer = {}
            # loop over state, action pairs of interest
            for state_action_pair, policies in self.open_loop_policies[layer].items():
                # compute the expected time, given the expected times of the lower layers
                timereward_layer[state_action_pair] = self.reward_li(layer, state_action_pair[0], policies)
            self.timerewards_layers.append(timereward_layer)
        self.timerewards_layers.reverse()

    def reward_li(self, layer, s_li, policies):
        # layer i is 1 layer above layer j
        R = 0
        nr_of_cases = 0
        # loop over policies, loop over relevant state variables
        for OL_policy in policies:
            for s_lj in OL_policy.keys():
                R += self.reward_pi_lj(layer + 1, s_li, list(s_lj), OL_policy[s_lj])
                nr_of_cases += 1
        R /= nr_of_cases
        return R

    # policy is a dict with key = xa_0, value = list of actions,
    def reward_pi_lj(self, layer_j, s_li, s_lj, OL_policy):
        R_pi_lj = 0
        gamma = 0.99
        for idx, a_lj in enumerate(OL_policy):
            R_pi_lj += gamma ** (idx) * self.reward_lj(layer_j, s_li, s_lj, a_lj)
            s_lj = self.next_state(s_lj, a_lj)
        return R_pi_lj

    def reward_lj(self, layer_j, s_li, s_lj, a_lj):
        if layer_j == self.nr_of_layers - 1:
            t_expected = 0
            if a_lj[0:3] == 'nav':
                t_expected = self.subroutine_actions[a_lj].t_expected[s_lj[0]]
            elif a_lj == 'pickup':
                if s_li[0] == s_li[1]:
                    t_expected = self.subroutine_actions[a_lj + '0'].t_expected[1][s_lj[0]]
                else:
                    print('uncovered case in reward_lj, s_lj = {}, a_lj = {}'.format(s_lj, a_lj))
            elif a_lj == 'release':
                t_expected = self.subroutine_actions[a_lj].t_expected[s_lj[0]]
            elif a_lj == 'look_around':
                if s_li[1] == 'none':
                    t_expected = self.subroutine_actions[a_lj].t_expected[0][s_lj[0]]
                else:
                    print('uncovered case in rewardl2, s_lj = {}, a_lj = {}'.format(s_lj, a_lj))
            else:
                print('uncovered case in reward_l2, a_lj = {}'.format(a_lj))
            return -t_expected
        else:
            if a_lj[0:3] == 'nav' and len(s_lj) > 1:
                key = (s_lj[0], a_lj)
            elif len(s_lj) == 1:
                key = (s_lj[0], a_lj)
            else:
                key = (tuple(s_lj), a_lj)
            return self.timerewards_layers[-1][key]

    # not generally doing what the name suggests. Just a help-function for O_pi_lj and reward_pi_lj
    def next_state(self, s_lj, a_lj):
        s_next = copy.copy(s_lj)
        if a_lj[0:3] == 'nav':
            ni, nj = auxiliary_functions.nav_action_get_n0(a_lj), auxiliary_functions.nav_action_get_n1(a_lj)
            if s_next[0] == ni:
                s_next[0] = nj
            elif s_next[0] == nj:
                s_next[0] = ni
        return s_next

    def compute_observation_probabilities_for_all_layers(self):
        # start with the second lowest layer and compute observation_probabilities with lowest layer models.
        for layer in range(self.nr_of_layers - 2, -1, -1):
            observation_probabilities_layer = {}
            # loop over state, action pairs of interest
            for state_action_pair, policies in self.open_loop_policies[layer].items():
                s_li = state_action_pair[0]
                a_li = state_action_pair[1]
                if a_li[0:3] == 'nav':
                    # loop over the xi values, set zi = xi and get observation probability
                    for zi in self.xa_values_all_layers[layer]:
                        key = tuple(list(state_action_pair) + [zi])
                        observation_probabilities_layer[key] = self.observation_model_li(layer, z_li=zi,
                                                                                         policies=policies)
                elif a_li[0:6] == 'pickup':
                    key = tuple(list(state_action_pair) + [s_li[0]])
                    observation_probabilities_layer[key] = self.observation_model_li(layer, z_li=s_li[0],
                                                                                     policies=policies)

                elif a_li == 'release':
                    continue  # note: if release is chosen in a not-goal-node, the item is released immediately -> prob. of 0
            self.observation_probabilities_layers.append(observation_probabilities_layer)
        self.observation_probabilities_layers.reverse()

    def observation_model_li(self, layer, z_li, policies):
        O = 0
        nr_of_cases = 0
        for OL_policy in policies:
            # loop over possible observations, given z_l1
            for z_lj in self.get_subobservations(z_li, layer):
                for key in OL_policy.keys():
                    O += self.O_pi_lj(layer + 1, z_lj, s_lj_0=key, remaining_OL_policy=OL_policy[key])
                    nr_of_cases += 1
        O /= nr_of_cases
        return O

    def O_pi_lj(self, layer_j, z_lj, s_lj_0, remaining_OL_policy):
        if len(remaining_OL_policy) == 0:
            return 0.0
        a_lj = remaining_OL_policy[0]
        s_lj = self.next_state(list(s_lj_0), a_lj)
        return self.observation_model_lj(layer_j, z_lj, a_lj, s_lj) + (
                    1 - self.observation_model_lj(layer_j, z_lj, a_lj, s_lj)) * \
               self.O_pi_lj(layer_j, z_lj, s_lj, remaining_OL_policy[1:])

    def observation_model_lj(self, layer_j, z_lj, a_lj, s_lj):
        xa_lj = s_lj[0]
        if layer_j == self.nr_of_layers - 1:
            if a_lj[0:3] == 'nav':
                if a_lj not in self.subroutine_actions_history[layer_j].values():
                    return self.subroutine_actions[a_lj].observation_probabilities[xa_lj][z_lj]
                else:
                    return 0.0
            elif a_lj[0:6] == 'pickup':
                if z_lj == xa_lj:
                    return 0.6  # simple assumption: if subroutine pickup0 runs and item1 is in the same node, prob. 0.6 of observing item1 during pickup of item0
                else:
                    return 0.0
            elif a_lj == 'look_around':
                if z_lj == xa_lj:
                    return self.subroutine_actions[a_lj].observation_probabilities[xa_lj]
                else:
                    return 0.0
            return 0.0
        else:
            if a_lj[0:6] == 'pickup' and xa_lj == z_lj:
                debug = True
            elif a_lj[0:6] == 'pickup' and s_lj[1] == 'none':
                debug = True
            xa_lj = s_lj[0]
            zero_prob = False
            if a_lj[0:3] == 'nav' and len(s_lj) > 1:
                key = (s_lj[0], a_lj, z_lj)
            elif len(s_lj) == 1:
                key = (s_lj[0], a_lj, z_lj)
            else:
                if (a_lj[0:6] == 'pickup' or a_lj == 'release') and xa_lj != z_lj:
                    zero_prob = True
                key = (tuple(s_lj), a_lj, z_lj)
            if not zero_prob:
                return self.observation_probabilities_layers[-1][key]
            else:
                return 0

    def read_open_loop_policies(self):
        open_loop_policies_layers = []
        for layer in range(self.nr_of_layers - 1):
            f = open(
                config.BASE_FOLDER_SIM + self.environment + '_environment/open_loop_policies/OL_policies_l{}.txt'.format(
                    1 + layer), "r")
            f.readline()
            open_loop_policies = {}
            for line in f:
                if line == '\n':
                    continue
                # if layer == 1:
                #     print(line)
                line_list = line.split(";")
                s = line_list[0].split()
                s[0] = int(s[0])
                for i in range(1, len(s)):
                    if s[i] != 'none':
                        s[i] = int(s[i])
                a = line_list[1]
                policy_str = line_list[2].split(".")
                policy = {}
                for dict_val in policy_str:
                    key_val_pair = dict_val.split(':')
                    key_list = [x for x in key_val_pair[0].split()]
                    for idx, el in enumerate(key_list):
                        if el == 'none':
                            continue
                        else:
                            key_list[idx] = int(el)
                    key = tuple(key_list)
                    # key = tuple([int(x) for x in key_val_pair[0].split()])
                    val = key_val_pair[1].split()
                    policy[key] = val
                if len(s) > 1:
                    key = (tuple(s), a)
                else:
                    key = (s[0], a)
                if key not in open_loop_policies.keys():
                    open_loop_policies[key] = [policy]
                else:
                    open_loop_policies[key] += [policy]
            open_loop_policies_layers.append(open_loop_policies)
        return open_loop_policies_layers

    # careful: the resulting state uses numbers for items not item_type as keys
    def deenumerate_state(self, s_li_nr, xa_values_li, x_items_values_li):
        s_li_copy = s_li_nr
        s = {'xa': xa_values_li[0]}
        key_values = ['xa']
        item_indices = list(x_items_values_li.keys())
        # order item_indices, starting with lowest
        item_indices.sort()
        for item_idx in item_indices:
            s[item_idx] = x_items_values_li[item_idx][0]
            key_values.append(item_idx)
        key_values_rev = copy.copy(key_values)
        key_values_rev.reverse()
        for var in key_values:
            nr_of_xi_values = 1
            for val in key_values_rev:
                if val == var:
                    break
                nr_of_xi_values *= len(x_items_values_li[val])
            s[var] = int(s_li_copy / nr_of_xi_values)
            s_li_copy = s_li_copy % nr_of_xi_values
            if var == 'xa':
                s['xa'] = xa_values_li[s['xa']]
            else:
                s[var] = x_items_values_li[var][s[var]]
        return s

    def update_carry(self, o_carry, item_x=None, item_y=None):
        for item_idx in range(self.nr_of_items):
            item_type = self.inverse_item_map(item_idx)
            if o_carry == self.items[item_idx].name:
                for layer in range(self.nr_of_layers):
                    self.s_layers[layer][item_type] = 'agent'
                self.carries = item_type
                # set belief grid to uniform with 0 belief
                self.belief.set_b_tot(item_nr=self.item_map[o_carry], total_belief=0.0)
                self.belief.fill_grid(type='uniform', item_nr=self.item_map[o_carry])
            elif o_carry != self.items[item_idx].name and self.s_layers[-1][item_type] == 'agent':
                if item_x is not None and item_y is not None:
                    item_node = self.pos_to_node(item_x, item_y, self.nr_of_layers - 1)
                else:
                    item_node = self.s_layers[-1]['xa']
                    item_x, item_y = self.x, self.y
                if item_node == self.items[item_idx].goal_nodes_layers[-1]:
                    for layer in range(self.nr_of_layers):
                        self.s_layers[layer][item_type] = 'goal'
                    self.carries = 'none'
                    self.belief.set_b_tot(item_idx, 0.0)
                else:
                    self.belief.set_b_tot(item_idx, 1.0)
                    self.belief.delete_belief_spots(item_nr=item_idx)
                    self.belief.add_belief_spot(mu_x=item_x, mu_y=item_y, prob=0.99, sigma=0.01, item_nr=item_idx)
                    for layer in range(self.nr_of_layers):
                        self.s_layers[layer][item_type] = -1
                    self.carries = 'none'

    def update_pose(self, o_x, o_y, o_theta):
        if not self.initialized:
            return
        # update agent configuration
        self.x, self.y, self.theta = o_x, o_y, o_theta
        self.u, self.v = self.grid.get_cell_indices_by_position(self.x, self.y)
        # update state variables
        for layer in range(self.nr_of_layers):
            self.s_layers[layer]['xa'] = self.pos_to_node(self.x, self.y, layer)

    def interpret_observations(self, o_cells):
        if not self.initialized:
            return
        # translate o_cells to item_nrs
        o_cells_nrs = []
        for o_idx in range(len(o_cells)):
            if o_cells[o_idx].obs_val in config.item_types:
                o_cells_nrs.append([o_cells[o_idx].x, o_cells[o_idx].y, self.item_map[o_cells[o_idx].obs_val]])
            else:
                o_cells_nrs.append([o_cells[o_idx].x, o_cells[o_idx].y, o_cells[o_idx].obs_val])
        if self.current_action_name_lN == 'look_around':
            self.subroutine_actions['look_around'].update_local_grid(self.s_layers[-1]['xa'], o_cells)
        self.belief.update_belief(observations=o_cells_nrs)

    def choose_action(self):
        # check if task is already solved
        task_finished = True
        for key in self.s_layers[0].keys():
            if key == 'xa':
                continue
            if self.s_layers[0][key] != 'goal':
                task_finished = False
        if task_finished:
            print('\n')
            print('***************************************************************************')
            print('*************************** task is finished ******************************')
            print('***************************************************************************')
            print('\n')
            basic_action = ('finished', (self.x, self.y, self.theta))
            return basic_action
        self.k += 1
        a_name_lN = self.current_action_name_lN
        if a_name_lN in self.subroutine_actions:
            a = self.subroutine_actions[a_name_lN]
            # execute subroutine
            goal_ref = a.subroutine(s=self.s_layers[-1], item_map=self.item_map, agent_x=self.x, agent_y=self.y,
                                    agent_theta=self.theta, grid=self.grid, nodegraph=self.nodegraphs_layers[-1],
                                    items=self.items, b0=self.b0_layers[-1])
        if a_name_lN == 'none' or goal_ref == 'finished':
            self.a_names = ['none']
            comp_time_per_layer = []
            for layer in range(self.nr_of_layers):
                comp_time_t0 = time.time()
                self.a_names.append(self.compute_policy(layer, self.a_names[-1]))
                self.subroutine_actions_history[layer][self.k] = self.a_names[-1]
                comp_time_t1 = time.time()
                comp_time = comp_time_t1 - comp_time_t0
                comp_time_per_layer.append(comp_time)
                self.computation_time_layers[layer] += comp_time
            self.computation_time_for_each_action.append(comp_time_per_layer)
            self.computation_time = sum(self.computation_time_layers)
            a_name_lN = self.a_names[-1]
            if a_name_lN[0:6] == 'pickup':
                debug = True
            # if it is a navigation action -> set observation prob. to 0
            if a_name_lN[0:3] == 'nav':
                n0, n1 = auxiliary_functions.nav_action_get_n0(a_name_lN), auxiliary_functions.nav_action_get_n1(
                    a_name_lN)
                n_dash = -1
                if n0 == self.s_layers[-1]['xa']:
                    n_dash = n1
                elif n1 == self.s_layers[-1]['xa']:
                    n_dash = n0
                for nj in self.xa_values_all_layers[-1]:
                    self.subroutine_actions[a_name_lN].observation_probabilities[n_dash][nj] = 0.0
                # update higher layer observation models
                self.compute_observation_probabilities_for_all_layers()
            self.current_action_name_lN = a_name_lN
            print('a = {}'.format(self.a_names))
            #     for i in range(len(self.items)):
            #         self.belief.draw_plot(agent_x=self.x, agent_y=self.y, item_nr=i, world_width=self.grid.total_width,
            #                           world_height=self.grid.total_height)
            # execute subroutine
            a = self.subroutine_actions[a_name_lN]
            goal_ref = a.subroutine(s=self.s_layers[-1], item_map=self.item_map, agent_x=self.x, agent_y=self.y,
                                    agent_theta=self.theta, grid=self.grid, nodegraph=self.nodegraphs_layers[-1],
                                    items=self.items, b0=self.b0_layers[-1])
            # delete all elements, to avoid tricky bugs where an old POMDP is accessed
            del self.POMDPs[:]

        return goal_ref

    # input argument is the layer number and the action of the layer on top of this layer
    def compute_policy(self, layer, a_name_top):
        xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li = \
            self.get_state_action_observation_sets(self.s_layers[layer], self.s_layers[layer - 1], layer, a_name_top,
                                                   len(self.items))
        if len(self.POMDPs) >= layer + 1:
            self.POMDPs[layer]['current'].set_variable_sets(xa_values_li, x_items_values_li, s_terminals_li,
                                                            z_values_li,
                                                            action_names_li)
        else:
            self.POMDPs.append({'current': POMDPProblem(xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                                        action_names_li)})
        if a_name_top[0:3] == 'nav' and len(s_terminals_li) > 1:
            top_layer = layer - 1
            # get next action above
            s_top_next = self.s_layers[top_layer].copy()
            n0_top, n1_top = auxiliary_functions.nav_action_get_n0(a_name_top), auxiliary_functions.nav_action_get_n1(
                a_name_top)
            if s_top_next['xa'] == n0_top:
                s_top_next['xa'] = n1_top
            elif s_top_next['xa'] == n1_top:
                s_top_next['xa'] = n0_top
            else:
                print('something went wrong when determining s_top_next for layer{}'.format(layer))
            # get a starting state for this layer in s_top_next
            s_copy = self.s_layers[layer].copy()
            s_copy['xa'] = self.get_subnodes(node=s_top_next['xa'], layer=top_layer)[0]
            POMDP_top_key = 'current'
            for s_terminal_top in self.POMDPs[top_layer]['current'].s_terminals:
                if s_top_next['xa'] == s_terminal_top['xa'][0]:
                    POMDP_top_key = 'next'
            if POMDP_top_key not in self.POMDPs[top_layer].keys():
                # solve first the next problem for the layer above
                top_top_layer = top_layer - 1
                s_top_top = self.s_layers[top_top_layer].copy()
                s_top_top['xa'] = self.var_lower_to_var_top(top_layer, s_top_next['xa'])
                POMDP_top_top_key = 'current'
                a_name_top_top_next = self.read_policy(s_top_top, top_top_layer,
                                                       self.POMDPs[top_top_layer][POMDP_top_top_key].xa_values,
                                                       self.POMDPs[top_top_layer][POMDP_top_top_key].x_items_values,
                                                       self.POMDPs[top_top_layer][POMDP_top_top_key].z_values,
                                                       self.POMDPs[top_top_layer][POMDP_top_top_key].action_names,
                                                       POMDP_top_top_key, alpha_already_computed=True)
                xa_values_top, x_items_values_top, s_terminals_top, action_names_top, z_values_top = \
                    self.get_state_action_observation_sets(s_top_next, s_top_top, top_layer, a_name_top_top_next,
                                                           len(self.items))
                self.POMDPs[top_layer][POMDP_top_key] = POMDPProblem(
                    xa_values_top, x_items_values_top, s_terminals_top, z_values_top, action_names_top)
                # create POMDPX file and solve with SARSOP
                self.create_POMDPX_file(s_top_next, top_layer, xa_values_top, x_items_values_top, s_terminals_top,
                                        action_names_top, z_values_top)
                # config.base_folder + "../sarsop-master/src/./pomdpsol
                os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
                    config.solving_precision, self.timeout_time,
                    self.file_names_input[layer], self.file_names_output[layer]))
                # get alpha vectors which are used for solving the actual problem
                alpha_vectors, alpha_vectors_attrib = self.read_alpha_vectors(self.file_names_output[top_layer],
                                                                              xa_values_top, x_items_values_top)
                self.POMDPs[top_layer][POMDP_top_key].set_alpha_vectors(alpha_vectors, alpha_vectors_attrib)

            a_name_top_next = self.read_policy(s_top_next, top_layer, self.POMDPs[top_layer][POMDP_top_key].xa_values,
                                               self.POMDPs[top_layer][POMDP_top_key].x_items_values,
                                               self.POMDPs[top_layer][POMDP_top_key].z_values,
                                               self.POMDPs[top_layer][POMDP_top_key].action_names, POMDP_top_key,
                                               alpha_already_computed=True)
            xa_values_li_next, x_items_values_li_next, s_terminals_li_next, action_names_li_next, z_values_li_next = \
                self.get_state_action_observation_sets(s_copy, s_top_next, layer, a_name_top_next,
                                                       len(self.items))
            # if item is relevant for the next POMDP problem, add current item state to s_terminals_li
            for item_key in x_items_values_li_next.keys():
                item_type = self.inverse_item_map(item_key)
                if self.s_layers[layer][item_type] == 'agent' or self.s_layers[layer][item_type] == 'goal':
                    for s_terminal_li in s_terminals_li:
                        s_terminal_li[item_key] = [self.s_layers[layer][item_type]]
            self.POMDPs[layer]['next'] = POMDPProblem(xa_values_li_next, x_items_values_li_next, s_terminals_li_next,
                                                      z_values_li_next, action_names_li_next)
            # create POMDPX file and solve with SARSOP
            self.create_POMDPX_file(s_copy, layer, xa_values_li_next, x_items_values_li_next, s_terminals_li_next,
                                    action_names_li_next, z_values_li_next)
            os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
                config.solving_precision, self.timeout_time,
                self.file_names_input[layer], self.file_names_output[layer]))
            # get alpha vectors which are used for solving the actual problem
            alpha_vectors, alpha_vectors_attrib = self.read_alpha_vectors(self.file_names_output[layer],
                                                                          xa_values_li_next, x_items_values_li_next)
            self.POMDPs[layer]['next'].set_alpha_vectors(alpha_vectors, alpha_vectors_attrib)
            # IMPORTANT: do not move this statement above create_POMDPX_file
            self.solve_terminal_states_problems = POMDP_top_key
        self.create_POMDPX_file(self.s_layers[layer], layer, xa_values_li, x_items_values_li, s_terminals_li,
                                action_names_li, z_values_li)
        # solve POMDP with SARSOP
        os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
            config.solving_precision, self.timeout_time,
            self.file_names_input[layer], self.file_names_output[layer]))
        a_name_li = self.read_policy(self.s_layers[layer], layer, xa_values_li, x_items_values_li, z_values_li,
                                     action_names_li, POMDP_key='current')
        self.solve_terminal_states_problems = 'no'
        return a_name_li

    def get_state_action_observation_sets(self, s_li, s_top, layer, a_name_top, nr_of_items):
        return [], {}, [], [], {}

    def get_state_action_observation_sets_l0(self, s_l0, nr_of_items):
        xa_values_li, x_items_values_li = [], {}  # x_items_value_l2 = {item_nr: [item_values]}
        s_terminals_li = []  # [[xa_t0, x0_t0, .., xn_t0], [xa_t1, ...], ...]
        z_values_li = {}  # key = item_nr, value = list of values
        xa_values_li = self.xa_values_all_layers[0]
        s_terminal = {'xa': ['*']}
        action_names_li = self.action_names_layers[0]
        for item_idx in range(nr_of_items):
            x_items_values_li[item_idx] = self.xi_values_all_layers[0]
            z_values_li[item_idx] = self.zi_values_all_layers[0]
            s_terminal[item_idx] = ['goal']
        s_terminals_li.append(s_terminal)
        return xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li

    def read_policy(self, s, layer, xa_values_li, x_items_values_li, z_values_li, action_names_li, POMDP_key='current',
                    alpha_already_computed=False):
        if not alpha_already_computed:
            alpha_vectors, alpha_vectors_attrib = self.read_alpha_vectors(self.file_names_output[layer], xa_values_li,
                                                                          x_items_values_li)
            self.POMDPs[layer][POMDP_key].set_alpha_vectors(alpha_vectors, alpha_vectors_attrib)
        else:
            alpha_vectors = self.POMDPs[layer][POMDP_key].alpha_vectors
            alpha_vectors_attrib = self.POMDPs[layer][POMDP_key].alpha_vectors_attrib
        # compute belief for every possible state
        belief_enumerated = self.get_belief_for_every_state(layer, s, self.b0_layers[layer], x_items_values_li,
                                                            z_values_li)
        # now find best alpha vector for current belief
        V_best = -100000
        a_best = 0
        sel = self.get_alpha_vectors_selection(s, xa_values_li, x_items_values_li, z_values_li, alpha_vectors_attrib)
        for idx, vector in enumerate(alpha_vectors[sel]):
            V_p = np.dot(vector, belief_enumerated)
            if V_p > V_best:
                V_best = V_p
                a_best = alpha_vectors_attrib[sel, 1][idx]
        print('(V, a)=({}, {})'.format(V_best, a_best))
        return action_names_li[a_best]

    def read_alpha_vectors(self, file_path, xa_values_li, x_items_values_li):
        tree = ET.parse(file_path)
        root = tree.getroot()
        # reshape alpha_vectors
        alpha_vectors = np.zeros((int(root[0].attrib['numVectors']), int(root[0].attrib['vectorLength'])))
        alpha_vectors_attrib = np.zeros((int(root[0].attrib['numVectors']), 2), dtype=int)
        for idx, alpha_vector in enumerate(root[0]):
            # get obs-value (i.e. xa)
            obsValue = int(alpha_vector.attrib['obsValue'])
            # get action of alpha vector
            action_nr = int(alpha_vector.attrib['action'])
            alpha_vectors_attrib[idx] = [obsValue, action_nr]
            values_str = alpha_vector.text
            values_str = values_str.lstrip()
            values_str = values_str.rstrip()
            values = [float(idx2) for idx2 in values_str.split(' ')]
            alpha_vectors[idx] = values
        return alpha_vectors, alpha_vectors_attrib

    def get_belief_for_every_state(self, layer, s_li, b0_li, x_items_values_li, z_values_li):
        b_of_s_li = []
        POMDP_item_indices = []
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys() and item_idx in z_values_li.keys():
                POMDP_item_indices.append(item_idx)
        for item_idx in POMDP_item_indices:
            item_type = self.inverse_item_map(item_idx)
            if item_type not in s_li.keys() and item_idx in s_li.keys():
                item_type = item_idx
            xj_values_li = x_items_values_li[item_idx]
            b_of_item_idx = []
            for xi_li in xj_values_li:
                if xi_li == 'agent':
                    b_of_item_idx.append(float(s_li[item_type] == 'agent'))
                elif xi_li == 'goal':
                    b_of_item_idx.append(float(s_li[item_type] == 'goal'))
                elif xi_li == 'not_here':
                    b_not_here = 0
                    for xi2 in self.xa_values_all_layers[layer]:
                        if xi2 not in xj_values_li:
                            b_xi2 = b0_li.get_aggregated_belief(node_nr=xi2)[item_idx]
                            # WARNING: if below if condition is changed, need to change it in
                            # "write_initial_belief_function_li", "get_belief_for_every_state",
                            # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                            if layer >= 0:
                                if b_xi2 > self.b0_threshold:
                                    b_xi2 = 1.0
                                elif b_xi2 < 1 - self.b0_threshold:
                                    b_xi2 = 0.0
                            b_not_here += b_xi2
                    b_of_item_idx.append(b_not_here)
                else:
                    b_xi_li = b0_li.get_aggregated_belief(node_nr=xi_li)[item_idx]
                    # WARNING: if below if condition is changed, need to change it in
                    # "write_initial_belief_function_li", "get_belief_for_every_state",
                    # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                    if layer >= 0:
                        if b_xi_li > self.b0_threshold:
                            b_xi_li = 1.0
                        elif b_xi_li < 1 - self.b0_threshold:
                            b_xi_li = 0.0
                    b_of_item_idx.append(b_xi_li)
            # normalize b_of_item_idx to 1.0
            b_sum = sum(b_of_item_idx)
            if b_sum > 0:
                b_of_item_idx = [b_of_item_idx[idx] / b_sum for idx in range(len(b_of_item_idx))]
            b_of_s_li.append(b_of_item_idx)
        nr_of_item_states = 1
        for item_idx in POMDP_item_indices:
            xj_values_li = x_items_values_li[item_idx]
            nr_of_item_states *= len(xj_values_li)
        item_indices = list(x_items_values_li.keys())
        item_indices.sort()
        belief_enumerated = np.zeros(nr_of_item_states)
        for s_i in range(len(belief_enumerated)):
            b = 1.0
            s_i_copy = s_i
            for item_idx in POMDP_item_indices:
                xj_values_li = x_items_values_li[item_idx]
                nr_of_xi_values = 1
                for idx in range(len(x_items_values_li.keys()) - 1, item_indices.index(item_idx), -1):
                    if item_indices[idx] not in POMDP_item_indices:
                        continue
                    nr_of_xi_values *= len(x_items_values_li[item_indices[idx]])
                xi_li = int(s_i_copy / nr_of_xi_values)
                s_i_copy = s_i_copy % nr_of_xi_values
                b *= b_of_s_li[POMDP_item_indices.index(item_idx)][xi_li]
            belief_enumerated[s_i] = b
        return belief_enumerated

    def get_alpha_vectors_selection(self, s_li, xa_values_li, x_items_values_li, z_values_li, alpha_vectors_attrib):
        MDP_items_indices = []
        sel = []
        for item_idx in x_items_values_li.keys():
            if item_idx not in z_values_li:
                MDP_items_indices.append(item_idx)
        if len(MDP_items_indices) == 0:
            sel = alpha_vectors_attrib[:, 0] == xa_values_li.index(s_li['xa'])
        else:
            MDP_items_indices.reverse()
            state_nr = 0
            multiplication_factor = 1
            for item_idx in MDP_items_indices:
                item_type = self.inverse_item_map(item_idx)
                if item_type not in s_li.keys() and item_idx in s_li.keys():
                    item_type = item_idx
                item_val = s_li[item_type]
                state_nr += x_items_values_li[item_idx].index(item_val) * multiplication_factor
                multiplication_factor *= len(x_items_values_li[item_idx])
            xa_li = s_li['xa']
            state_nr += xa_values_li.index(xa_li) * multiplication_factor
            sel = alpha_vectors_attrib[:, 0] == state_nr
        return sel

    def create_POMDPX_file(self, s, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li,
                           z_values_li, a_top='none'):
        # create pomdpx file
        f = open(self.file_names_input[layer], "w+")
        # write header
        f.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n")
        f.write("<pomdpx version=\"1.0\" id=\"simplifiedMDP\"\n")
        f.write("\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
        f.write("\txsi:noNamespaceSchemaLocation=\"pomdpx.xsd\">\n")

        # write Description tag
        f.write("\t<Description> MultiScale POMDP Layer {}\n".format(layer))
        f.write("\t</Description>\n")
        # write Discount tag
        f.write("\t<Discount> {} </Discount>\n".format(auxiliary_functions.get_discount_factor(self.environment)))
        # write Variable Tag
        self.write_variables_li(f, xa_values_li, x_items_values_li, action_names_li, z_values_li)
        # write InitialStateBelief tag
        self.write_initial_belief_function_li(f, s, layer, xa_values_li, x_items_values_li)
        # write StateTransitionFunction tag
        self.write_transition_function_li(f, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li)
        # write ObsFunction Tag
        if len(z_values_li.keys()) > 0:
            self.write_observation_function_li(f, layer, xa_values_li, x_items_values_li, z_values_li, action_names_li)
        # write RewardFunction Tag
        self.write_reward_function_li(f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                      action_names_li, a_top)

        # close pomdpx tag
        f.write("</pomdpx>")
        f.close()

    def write_variables_li(self, f, xa_values_li, x_items_values_li, action_names_li, z_values_li):
        f.write("\t<Variable>\n")
        # State Variables
        # agent position
        f.write("\t\t<StateVar vnamePrev=\"xa_0\" vnameCurr=\"xa_1\" fullyObs=\"true\">\n")
        state_string = ''
        for xa in xa_values_li:
            state_string += ' s{}'.format(xa)
        f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(state_string))
        f.write("\t\t</StateVar>\n")
        # items
        for item_idx, xi_values in x_items_values_li.items():
            state_string = ''
            for xi in xi_values:
                if xi == 'agent' or xi == 'goal' or xi == 'not_here':
                    state_string += ' {}'.format(xi)
                else:
                    state_string += ' s{}'.format(xi)
            is_item_fully_obs = 'false'
            if item_idx not in z_values_li.keys():
                is_item_fully_obs = 'true'
            f.write("\t\t<StateVar vnamePrev=\"x{}_0\" vnameCurr=\"x{}_1\" fullyObs=\"".format(item_idx,
                                                                                               item_idx) + is_item_fully_obs + "\">\n")
            f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(state_string))
            f.write("\t\t</StateVar>\n")
        # Observation Variables
        for item_idx, zi_values in z_values_li.items():
            value_table = ''
            for zi in zi_values:
                if zi == 'no' or zi == 'agent':
                    value_table += ' {}'.format(zi)
                else:
                    value_table += ' o{}'.format(zi)
            f.write("\t\t<ObsVar vname=\"z{}\">\n".format(item_idx))
            f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(value_table))
            f.write("\t\t</ObsVar>\n")
        # Action Variables
        f.write("\t\t<ActionVar vname=\"action_agent\">\n")
        action_string = 'a' + action_names_li[0]
        for a in range(1, len(action_names_li)):
            action_string += ' a{}'.format(action_names_li[a])
        f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(action_string))
        f.write("\t\t</ActionVar>\n")
        # Reward Variables
        f.write("\t\t<RewardVar vname=\"reward_time\" />\n")
        f.write("\t\t<RewardVar vname=\"reward_subtask\" />\n")
        f.write("\t\t<RewardVar vname=\"reward_task\" />\n")
        f.write("\t\t<RewardVar vname=\"reward_terminal\" />\n")
        f.write("\t</Variable>\n")

    def write_initial_belief_function_li(self, f, s_li, layer, xa_values_li, x_items_values_li):
        f.write("\t<InitialStateBelief>\n")
        f.write("\t\t<CondProb>\n")
        # agent position xa
        belief_string = ''
        for xa in xa_values_li:
            if xa == s_li['xa']:
                belief_string += ' 1.0'
            else:
                belief_string += ' 0.0'
        f.write("\t\t\t<Var>xa_0</Var>\n")
        f.write("\t\t\t<Parent>null</Parent>\n")
        f.write("\t\t\t<Parameter type=\"TBL\">\n")
        self.write_entry_probtable(f, instance_string='-', prob_table_string=belief_string)
        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</CondProb>\n")
        # items
        b0_layers = copy.deepcopy(self.b0_layers)
        for item_idx, xi_values in x_items_values_li.items():
            item_type = self.inverse_item_map(item_idx)
            belief_vec = []
            belief_of_nodes = {}
            set_to_one = False
            for xa in xa_values_li:
                belief_of_nodes[xa] = b0_layers[layer].get_aggregated_belief(node_nr=xa)[item_idx]
                # WARNING: if below if confition is changed, need to change it in
                # "write_initial_belief_function_li", "get_belief_for_every_state",
                # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                if layer >= 0:  # not sure if needed for all layers
                    if belief_of_nodes[xa] > self.b0_threshold:
                        set_to_one = True
                    elif belief_of_nodes[xa] < 1 - self.b0_threshold:
                        belief_of_nodes[xa] = 0.0
                if set_to_one:
                    for key in belief_of_nodes:
                        if belief_of_nodes[key] > self.b0_threshold:
                            belief_of_nodes[key] = 1.0
                        else:
                            belief_of_nodes[key] = 0.0
            for xi in xi_values:
                if xi == 'agent' and s_li[item_type] == 'agent':
                    belief_vec.append(1.0)
                elif xi == 'agent' and s_li[item_type] != 'agent':
                    belief_vec.append(0.0)
                elif xi == 'goal' and s_li[item_type] == 'goal':
                    belief_vec.append(1.0)
                elif xi == 'goal' and s_li[item_type] != 'goal':
                    belief_vec.append(0.0)
                else:
                    if xi == 'not_here':
                        prob = 0
                        if not set_to_one:
                            for xa in self.xa_values_all_layers[layer]:
                                if xa not in xi_values:
                                    prob += b0_layers[layer].get_aggregated_belief(node_nr=xa)[item_idx]
                    else:
                        prob = belief_of_nodes[xi]
                    belief_vec.append(prob)
            # normalize belief_vec
            belief_sum = sum(belief_vec)
            belief_vec = [val / belief_sum for val in belief_vec]
            # convert belief_vec to belief_string
            belief_vec_str = ['{:.20f}'.format(val) for val in belief_vec]
            belief_string = ' '.join(belief_vec_str)
            f.write("\t\t<CondProb>\n")
            f.write("\t\t\t<Var>x{}_0</Var>\n".format(item_idx))
            f.write("\t\t\t<Parent>null</Parent>\n")
            f.write("\t\t\t<Parameter type=\"TBL\">\n")
            self.write_entry_probtable(f, instance_string='-', prob_table_string=belief_string)
            f.write("\t\t\t</Parameter>\n")
            f.write("\t\t</CondProb>\n")
        f.write("\t</InitialStateBelief>\n")

    def write_transition_function_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li):
        f.write("\t<StateTransitionFunction>\n")
        self.transition_function_xa_li(f, xa_values_li, x_items_values_li, s_terminals_li, action_names_li)
        for item_idx, xi_values_li in x_items_values_li.items():
            self.transition_function_xi_li(f, layer, item_idx, xa_values_li, x_items_values_li, s_terminals_li,
                                           action_names_li)
        f.write("\t</StateTransitionFunction>\n")

    def transition_function_xa_li(self, f, xa_values_li, x_items_values_li, s_terminals_li, action_names_li):
        f.write("\t\t<CondProb>\n")
        parent_string = 'action_agent xa_0'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx)
        f.write("\t\t\t<Var>xa_1</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        for a in action_names_li:
            if a[0:3] == 'nav':
                i, j = auxiliary_functions.nav_action_get_n0(a), auxiliary_functions.nav_action_get_n1(a)
                prob_table_string = ''
                for xa_0 in xa_values_li:
                    for xa_1 in xa_values_li:
                        if xa_0 == i and xa_1 == j:
                            prob_table_string += ' 1.0'
                        elif xa_0 == j and xa_1 == i:
                            prob_table_string += ' 1.0'
                        elif xa_0 == xa_1 and xa_0 != i and xa_1 != i and xa_0 != j and xa_1 != j:
                            prob_table_string += ' 1.0'
                        else:
                            prob_table_string += ' 0.0'
                instance_string = 'a{} -'.format(a)
                for item_idx in x_items_values_li.keys():
                    instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string)
            elif a[0:6] == 'pickup':
                instance_string = 'a{} -'.format(a)
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
            elif a == 'release':
                instance_string = 'a{} -'.format(a)
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
            elif a == 'look_around':
                instance_string = 'a{} -'.format(a)
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
        # terminal states   ############################################################################################
        for s_terminal in s_terminals_li:
            instance_string = '*'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none' or (var_key != 'xa' and var_key not in x_items_values_li.keys()):
                        continue
                    elif (var_t[0] == '*' or len(var_t) > 1) and var_key == 'xa':
                        instance_string += ' -'
                    elif (var_t[0] == '*' or len(var_t) > 1) and var_key != 'xa':
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            instance_string += ' -'
            prob_table_string = ''
            if s_terminal['xa'][0] == '*':  # or len(s_terminal['xa']) > 1
                prob_table_string = 'identity'
            else:
                for xa_1 in xa_values_li:
                    if len(s_terminal['xa']) == 1:
                        if xa_1 == s_terminal['xa'][0]:
                            prob_table_string += ' 1.0'
                        else:
                            prob_table_string += ' 0.0'
                    else:
                        for xa_1_dash in xa_values_li:
                            if xa_1 == xa_1_dash:
                                prob_table_string += ' 1.0'
                            else:
                                prob_table_string += ' 0.0'
            self.write_entry_probtable(f, instance_string, prob_table_string)
        # illegal states ###############################################################################################
        # are there multiple items that have the value agent?
        agent_counter = 0
        index_list = []
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                xi_values_li = x_items_values_li[item_idx]
                if 'agent' in xi_values_li:
                    agent_counter += 1
                    index_list.append(item_idx)
        if agent_counter > 1:
            ag1 = index_list[0]
            ag2 = index_list[1]
            for n in range(len(index_list)):
                instance_string = '* -'
                # translate ag1 ag2 into an instance
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        if item_idx == ag1 or item_idx == ag2:
                            instance_string += ' agent'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
                if ag2 == index_list[-1]:
                    ag1 = index_list[index_list.index(ag1) + 1]
                    if ag1 == index_list[-1]:
                        break
                    else:
                        ag2 = index_list[index_list.index(ag1) + 1]
                else:
                    ag2 = index_list[index_list.index(ag2) + 1]

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</CondProb>\n")

    def transition_function_xi_li(self, f, layer, item_nr, xa_values_li, x_items_values_li, s_terminals_li,
                                  action_names_li):
        xj_values_li = x_items_values_li[item_nr]
        parent_string = 'action_agent xa_0'
        for item_idx2 in range(self.nr_of_items):
            if item_idx2 in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx2)
        f.write("\t\t<CondProb>\n")
        f.write("\t\t\t<Var>x{}_1</Var>\n".format(item_nr))
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")

        # action= nav(ij), xi_0 == xi_1 -> T = 1
        for a in action_names_li:
            if a[0:3] == 'nav':
                instance_string = 'a{} *'.format(a)
                for item_idx2 in range(self.nr_of_items):
                    if item_idx2 in x_items_values_li.keys():
                        if item_idx2 == item_nr:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
            elif a[0:6] == 'pickup':
                i = auxiliary_functions.pickup_get_i(a)
                if i != item_nr:
                    # if agent picks up an other item, this item is remaining in same state
                    instance_string = 'a{} *'.format(a)
                    for item_idx2 in range(self.nr_of_items):
                        if item_idx2 in x_items_values_li.keys():
                            if item_idx2 == item_nr:
                                instance_string += ' -'
                            else:
                                instance_string += ' *'
                    instance_string += ' -'
                    self.write_entry_probtable(f, instance_string, prob_table_string='identity')
                # special case if i == k, P = 1.0 to transition to agent
                else:
                    self.pickup_transition_prob_li(f, layer, a, i, xa_values_li, x_items_values_li)
            elif a == 'release':
                # over-general rule xi_1 = xi_0
                instance_string = 'arelease *'
                for item_idx2 in range(self.nr_of_items):
                    if item_idx2 in x_items_values_li.keys():
                        if item_idx2 == item_nr:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
                # if xi_0 = agent -> xi_1 = xa_0
                if 'agent' not in xj_values_li:
                    continue
                instance_string = 'arelease -'
                for item_idx2 in range(self.nr_of_items):
                    if item_idx2 in x_items_values_li.keys():
                        if item_idx2 == item_nr:
                            instance_string += ' agent'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                prob_table = ['0.0'] * (len(xa_values_li) * len(xj_values_li))
                prob_table_string = ''
                instance = [xa_values_li[0], xj_values_li[0]]
                for idx, val in enumerate(prob_table):
                    if instance[0] == self.items[item_nr].goal_nodes_layers[layer] and instance[1] == 'goal':
                        prob_table_string += ' 1.0'
                    elif instance[0] != self.items[item_nr].goal_nodes_layers[layer] and instance[0] == instance[1]:
                        prob_table_string += ' 1.0'
                    else:
                        prob_table_string += ' 0.0'
                    # increase state by 1
                    for k in range(len(instance) - 1, -1, -1):
                        if instance[k] == xj_values_li[-1]:
                            instance[k] = xj_values_li[0]
                        else:
                            if k == 0:
                                if idx == len(prob_table) - 1:
                                    break
                                instance[k] = xa_values_li[xa_values_li.index(instance[k]) + 1]
                            else:
                                instance[k] = xj_values_li[xj_values_li.index(instance[k]) + 1]
                            break
                self.write_entry_probtable(f, instance_string, prob_table_string)
            elif a == 'look_around':
                instance_string = 'a{} *'.format(a)
                for item_idx2 in range(self.nr_of_items):
                    if item_idx2 in x_items_values_li.keys():
                        if item_idx2 == item_nr:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
        # terminal states   ########################################################################################
        for s_terminal in s_terminals_li:
            instance_string = '*'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none' or (var_key != 'xa' and var_key not in x_items_values_li.keys()):
                        continue
                    elif (var_t[0] == '*' or len(var_t) > 1) and var_key == item_nr:
                        instance_string += ' -'
                    elif (var_t[0] == '*' or len(var_t) > 1) and var_key != item_nr:
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            instance_string += ' -'
            prob_table_string = ''
            if s_terminal[item_nr] == ['*']:
                prob_table_string = 'identity'
            else:
                for xj_1 in xj_values_li:
                    if len(s_terminal[item_nr]) == 1:
                        if xj_1 == s_terminal[item_nr][0]:
                            prob_table_string += ' 1.0'
                        else:
                            prob_table_string += ' 0.0'
                    else:
                        for xj_1_dash in xj_values_li:
                            if xj_1 == xj_1_dash:
                                prob_table_string += ' 1.0'
                            else:
                                prob_table_string += ' 0.0'
            self.write_entry_probtable(f, instance_string, prob_table_string)
        # illegal states   ###########################################################################################
        # are there multiple items that have the value agent?
        agent_counter = 0
        index_list = []
        for item_idx2 in range(self.nr_of_items):
            if item_idx2 in x_items_values_li.keys():
                xk_values_li = x_items_values_li[item_idx2]
                if 'agent' in xk_values_li:
                    agent_counter += 1
                    index_list.append(item_idx2)
        if agent_counter > 1:
            ag1 = index_list[0]
            ag2 = index_list[1]
            for n in range(len(index_list)):
                instance_string = '* *'
                # translate ag1 ag2 into an instance
                for item_idx2 in range(self.nr_of_items):
                    if item_idx2 in x_items_values_li.keys():
                        if item_idx2 == ag1 or item_idx2 == ag2:
                            instance_string += ' agent'
                        elif item_idx2 == item_nr:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                prob_table_string = 'identity'
                if instance_string.count('-') == 1:
                    prob_table_string = ''
                    for xi in xj_values_li:
                        if xi == 'agent':
                            prob_table_string += ' 1.0'
                        else:
                            prob_table_string += ' 0.0'
                self.write_entry_probtable(f, instance_string, prob_table_string)
                if ag2 == index_list[-1]:
                    ag1 = index_list[index_list.index(ag1) + 1]
                    if ag1 == index_list[-1]:
                        break
                    else:
                        ag2 = index_list[index_list.index(ag1) + 1]
                else:
                    ag2 = index_list[index_list.index(ag2) + 1]

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</CondProb>\n")

    def pickup_transition_prob_li(self, f, layer, a, item_i, xa_values_li, x_items_values_li):
        # rule 1: xa=-, x0 = *, ..., xi = -, xi' = -    ################################################################
        instance_string = 'a{} -'.format(a)
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                if item_idx == item_i:
                    instance_string += ' -'
                else:
                    instance_string += ' *'
        instance_string += ' -'
        # prob table
        prob_table_string = ''
        xj_values_li = x_items_values_li[item_i]
        instance = [xa_values_li[0], xj_values_li[0], xj_values_li[0]]
        nr_of_states = len(xa_values_li) * len(xj_values_li) ** 2
        for state_nr in range(nr_of_states):
            if instance[0] == instance[1] and instance[2] == 'agent':
                prob_table_string += ' 1.0'
            elif instance[0] == self.items[item_i].goal_nodes_layers[layer] and instance[1] == 'goal' and instance[
                2] == 'agent':
                prob_table_string += ' 1.0'
            elif instance[0] == self.items[item_i].goal_nodes_layers[layer] and instance[1] == 'goal' and instance[1] == \
                    instance[2]:
                prob_table_string += ' 0.0'
            elif instance[0] != instance[1] and instance[1] == instance[2]:
                prob_table_string += ' 1.0'
            else:
                prob_table_string += ' 0.0'
            # increase state by 1
            for k in range(len(instance) - 1, -1, -1):
                if instance[k] == xj_values_li[-1]:
                    instance[k] = xj_values_li[0]
                else:
                    instance[k] = xj_values_li[xj_values_li.index(instance[k]) + 1]
                    break
        self.write_entry_probtable(f, instance_string, prob_table_string)
        # rule 2: some other xj is already 'agent' #####################################################################
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                xk_values_li = x_items_values_li[item_idx]
                if item_idx == item_i or 'agent' not in xk_values_li:
                    continue
                instance_string = 'apickup{} *'.format(item_i)
                for item_idx2 in range(self.nr_of_items):
                    if item_idx2 in x_items_values_li.keys():

                        if item_idx == item_idx2:
                            instance_string += ' agent'
                        elif item_idx2 == item_i:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')

    def write_observation_function_li(self, f, layer, xa_values_li, x_items_values_li, z_values_li, action_names_li):
        f.write("\t<ObsFunction>\n")
        # loop over observation variables
        for z_idx in range(self.nr_of_items):
            if z_idx in z_values_li.keys():
                zj_values_li = z_values_li[z_idx]
                f.write("\t\t<CondProb>\n")
                parent_string = 'action_agent xa_1'
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        parent_string += ' x{}_1'.format(item_idx)
                f.write("\t\t\t<Var>z{}</Var>\n".format(z_idx))
                f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
                f.write("\t\t\t<Parameter type = \"TBL\">\n")
                # for each action a rule is written
                for a in action_names_li:
                    if a[0:3] == 'nav':
                        ni, nj = auxiliary_functions.nav_action_get_n0(a), auxiliary_functions.nav_action_get_n1(a)
                        # general rule: prob. of observing zi = no is 1 for all states
                        instance_string = 'a{} *'.format(a)
                        for item_idx in range(self.nr_of_items):
                            if item_idx in x_items_values_li.keys():
                                instance_string += ' *'
                        instance_string += ' -'
                        prob_table_string = ''  # prob. 1 of observing "no"
                        for zi in zj_values_li:
                            if zi == 'no':
                                prob_table_string += '1.0'
                            else:
                                prob_table_string += ' 0.0'
                        self.write_entry_probtable(f, instance_string, prob_table_string)
                        # overwrite rule with exception: if xa=i, xi= j
                        self.obs_function_nav_li(f, layer, a, n_dash=ni, z_idx=z_idx,
                                                 x_items_values_li=x_items_values_li,
                                                 zj_values_li=zj_values_li)
                        self.obs_function_nav_li(f, layer, a, n_dash=nj, z_idx=z_idx,
                                                 x_items_values_li=x_items_values_li,
                                                 zj_values_li=zj_values_li)
                    elif a[0:6] == 'pickup':
                        if layer == self.nr_of_layers - 1:
                            instance_string = 'a{} *'.format(a)
                            for item_idx in range(self.nr_of_items):
                                if item_idx in x_items_values_li.keys():
                                    instance_string += ' *'
                            instance_string += ' no'
                            prob_table_string = '1.0'
                            self.write_entry_probtable(f, instance_string, prob_table_string)
                        else:
                            self.obs_function_pickup_li(f, layer, a, z_idx, xa_values_li, x_items_values_li,
                                                        zj_values_li)
                    elif a == 'release':
                        instance_string = 'a{} *'.format(a)
                        for item_idx in range(self.nr_of_items):
                            if item_idx in x_items_values_li.keys():
                                instance_string += ' *'
                        instance_string += ' no'
                        prob_table_string = '1.0'
                        self.write_entry_probtable(f, instance_string, prob_table_string)
                    elif a == 'look_around':
                        instance_string = 'a{} -'.format(a)
                        for item_idx in range(self.nr_of_items):
                            if item_idx in x_items_values_li.keys():
                                if item_idx == z_idx:
                                    instance_string += ' -'
                                else:
                                    instance_string += ' *'
                        instance_string += ' -'
                        prob_table_string = ''
                        for xa_li in xa_values_li:
                            for xj_li in x_items_values_li[z_idx]:
                                for zj_li in zj_values_li:
                                    if xa_li == xj_li and xa_li == zj_li:
                                        value = '{}'.format(self.subroutine_actions[a].observation_probabilities[xa_li])
                                    elif xa_li == xj_li and zj_li == 'no' and all(
                                            xa in zj_values_li for xa in xa_values_li):
                                        value = '{}'.format(
                                            1 - self.subroutine_actions[a].observation_probabilities[xa_li])
                                    elif xj_li == 'agent' and zj_li == 'agent':
                                        value = '1.0'
                                    elif zj_li == 'no':
                                        value = '1.0'
                                    else:
                                        value = '0.0'
                                    prob_table_string += ' {}'.format(value)
                        self.write_entry_probtable(f, instance_string, prob_table_string)
                # for all actions: if xk = agent, zk is 'agent' with P=1.0
                if 'agent' in x_items_values_li[z_idx] and 'agent' in z_values_li[z_idx]:
                    instance_string = '* *'
                    for item_idx in range(self.nr_of_items):
                        if item_idx in x_items_values_li.keys():
                            if item_idx == z_idx:
                                instance_string += ' agent'
                            else:
                                instance_string += ' *'
                    instance_string += ' -'
                    prob_table_string = ''
                    for zi in zj_values_li:
                        if zi == 'agent':
                            prob_table_string += ' 1.0'
                        else:
                            prob_table_string += ' 0.0'
                    self.write_entry_probtable(f, instance_string, prob_table_string)

                # close variable
                f.write("\t\t\t</Parameter>\n")
                f.write("\t\t</CondProb>\n")
        f.write("\t</ObsFunction>\n")

    def obs_function_nav_li(self, f, layer, a, n_dash, z_idx, x_items_values_li, zj_values_li):
        n_0 = auxiliary_functions.nav_action_get_n0(a)
        if n_0 == n_dash:
            n_0 = auxiliary_functions.nav_action_get_n1(a)
        instance_string = 'a{} s{}'.format(a, n_dash)
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                if item_idx == z_idx:
                    instance_string += ' -'
                else:
                    instance_string += ' *'
        instance_string += ' -'
        prob_table_string = ''
        # loop over all item positions:
        for xi in x_items_values_li[z_idx]:
            # loop over all observation values
            for zi in zj_values_li:
                if xi == zi and xi != 'agent':
                    if layer == self.nr_of_layers - 1:
                        value = '{}'.format(self.subroutine_actions[a].observation_probabilities[n_dash][xi])
                    else:
                        value = '{}'.format(self.observation_probabilities_layers[layer][(n_0, a, zi)])
                elif zi == 'no' and xi != 'agent' and xi != 'goal' and xi != 'not_here':
                    if layer == self.nr_of_layers - 1:
                        value = '{}'.format(1 - self.subroutine_actions[a].observation_probabilities[n_dash][xi])
                    else:
                        value = '{}'.format(1 - self.observation_probabilities_layers[layer][(n_0, a, xi)])
                elif zi == 'no' and xi == 'not_here':
                    value = '1.0'
                elif xi == 'agent' and zi == 'agent':
                    value = '1.0'
                elif xi == 'goal' and zi == 'no':
                    value = '1.0'
                else:
                    value = '0.0'
                prob_table_string += ' ' + value
        self.write_entry_probtable(f, instance_string, prob_table_string)

    def obs_function_pickup_li(self, f, layer, a, z_idx, xa_values_li, x_items_values_li, zj_values_li):
        instance_string = 'a{} -'.format(a)
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                instance_string += ' -'  # consider all the values
        instance_string += ' -'
        prob_table_string = ''
        nr_of_states = len(xa_values_li) * np.prod(
            [len(x_items_values_li[item_idx]) for item_idx in x_items_values_li.keys()])
        # loop over xa, item combinations
        item_i = auxiliary_functions.pickup_get_i(a)
        for s_li_nr in range(nr_of_states):
            s_li = self.deenumerate_state(s_li_nr, xa_values_li, x_items_values_li)
            xa_li = s_li['xa']
            idx_it_i = item_i
            idx_it_z = z_idx
            # loop over all observation values
            for zi in zj_values_li:
                value = ''
                if item_i == z_idx:
                    # either the obs is 'no' or 'agent'
                    if zi == 'agent' and s_li[idx_it_i] == 'agent':
                        value = '1.0'
                    elif zi == 'no':
                        value = '1.0'
                    else:
                        value = '0.0'
                else:
                    # the following if elif statement is necessary!
                    # If the top layer decides to pickup an item when already carrying one, then
                    # the lower layers POMDP is ill-defined
                    if s_li[idx_it_i] == 'agent' and zi == 'no':
                        value = '1.0'
                    elif s_li[idx_it_i] == 'agent' and zi != 'no':
                        value = '0.0'
                    elif xa_li == s_li[idx_it_z]:
                        if s_li[idx_it_z] == zi:
                            value = '{}'.format(self.observation_probabilities_layers[layer][
                                                    (tuple([xa_li, 'none']), 'pickup', zi)])
                        elif zi == 'no':
                            value = '{}'.format(1 - self.observation_probabilities_layers[layer][
                                (tuple([xa_li, 'none']), 'pickup', xa_li)])
                        else:
                            value = '0.0'
                    elif s_li[idx_it_z] == 'agent' and zi == 'agent':
                        value = '1.0'
                    elif xa_li != s_li[idx_it_z] and zi == 'no':
                        value = '1.0'
                    elif s_li[idx_it_z] == 'goal' and zi == 'no':
                        value = '1.0'
                    else:
                        value = '0.0'
                prob_table_string += ' ' + value
        self.write_entry_probtable(f, instance_string, prob_table_string)

    def write_reward_function_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                 action_names_li, a_top='none'):
        f.write("\t<RewardFunction>\n")
        self.write_reward_time_li(f, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li)
        self.write_reward_subtask_li(f, x_items_values_li, s_terminals_li, action_names_li)
        self.write_reward_task_li(f, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li)
        self.write_reward_terminal_li(f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li, a_top)
        f.write("\t</RewardFunction>\n")

    def write_reward_time_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li):
        parent_string = 'action_agent xa_0'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_time</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        for a in action_names_li:
            penalty = 1.0
            if a[0:3] == 'nav':
                instance_string = 'a{} -'.format(a)
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                value_table_string = ''
                for xa in xa_values_li:
                    if xa == auxiliary_functions.nav_action_get_n0(a) or xa == auxiliary_functions.nav_action_get_n1(a):
                        if layer == self.nr_of_layers - 1:
                            R = -self.subroutine_actions[a].t_expected[xa]
                        else:
                            R = self.timerewards_layers[layer][(xa, a)]
                        value_table_string += ' {}'.format(R * penalty)
                    else:
                        value_table_string += ' {}'.format(-1000.0)
                self.write_entry_valuetable(f, instance_string, value_table_string)
            elif a == 'look_around':
                instance_string = 'a{} -'.format(a)
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' -'
                value_table_string = ''
                nr_of_states = len(xa_values_li)
                for item_idx, xj_values_li in x_items_values_li.items():
                    nr_of_states *= len(xj_values_li)
                for state_nr in range(nr_of_states):
                    state = self.deenumerate_state(s_li_nr=state_nr, xa_values_li=xa_values_li,
                                                   x_items_values_li=x_items_values_li)
                    is_item_present = False
                    key_list = ['xa'] + list(range(0, self.nr_of_items))
                    for key in key_list:
                        if key in state.keys():
                            if key == 'xa':
                                continue
                            if state['xa'] == state[key]:
                                value_table_string += ' -{}'.format(
                                    penalty * self.subroutine_actions[a].t_expected[1][state['xa']])
                                is_item_present = True
                                break
                    if not is_item_present:
                        value_table_string += ' -{}'.format(
                            penalty * self.subroutine_actions[a].t_expected[0][state['xa']])
                self.write_entry_valuetable(f, instance_string, value_table_string)
            elif a[0:6] == 'pickup':
                i = auxiliary_functions.pickup_get_i(a)
                instance_string = 'a{} -'.format(a)
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        if item_idx == i:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                value_table_string = ''
                s_li = [0, 'none']
                for xa in xa_values_li:
                    s_li[0] = xa
                    for xi in x_items_values_li[i]:
                        if xa == xi or (xa == self.items[i].goal_nodes_layers[layer] and xi == 'goal'):
                            if xi == 'goal':
                                s_li[1] = self.items[i].goal_nodes_layers[layer]
                            else:
                                s_li[1] = xi
                            if layer == self.nr_of_layers - 1:
                                R = -self.subroutine_actions[a].t_expected[1][xa]
                            else:
                                R = self.timerewards_layers[layer][(tuple(s_li), a[0:6])]
                            value_table_string += ' {}'.format(penalty * R)
                        elif xi != 'agent':
                            s_li[1] = 'none'
                            if layer == self.nr_of_layers - 1:
                                R = -self.subroutine_actions[a].t_expected[0][xa]
                            else:
                                R = self.timerewards_layers[layer][(tuple(s_li), a[0:6])]
                            value_table_string += ' {}'.format(R * penalty)
                        elif xi == 'agent':
                            value_table_string += ' {}'.format(0)
                self.write_entry_valuetable(f, instance_string, value_table_string)
            elif a == 'release':
                # general rule: negative reward for illegal release action
                instance_string = 'arelease *'
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                self.write_entry_valuetable(f, instance_string,
                                            value_table_string='-{}'.format(config.robot_release_time))
                # release when a variable is actually 'agent'
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        if 'agent' not in x_items_values_li[item_idx]:
                            continue
                        instance_string = 'arelease -'
                        for k in x_items_values_li.keys():
                            if k == item_idx:
                                instance_string += ' agent'
                            else:
                                instance_string += ' *'
                        value_table_string = ''
                        for xa in xa_values_li:
                            if xa == self.items[item_idx].goal_nodes_layers[layer]:
                                if layer == self.nr_of_layers - 1:
                                    R = -self.subroutine_actions[a].t_expected[xa]
                                else:
                                    R = self.timerewards_layers[layer][(xa, a)]
                                value_table_string += ' {}'.format(R * penalty)
                            else:
                                value_table_string += ' -{}'.format(config.robot_release_time)
                        self.write_entry_valuetable(f, instance_string, value_table_string)
        # reward 0 for terminal states
        for s_terminal in s_terminals_li:
            instance_string = '*'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none' or (var_key != 'xa' and var_key not in x_items_values_li.keys()):
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')

        # reward 0 for illegal states
        self.write_reward_illegal_states_li(f, x_items_values_li, current_states=False)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def write_reward_subtask_li(self, f, x_items_values_li, s_terminals_li, action_names_li):
        parent_string = 'action_agent xa_0'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx)
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_1'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_subtask</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        # for each item, picking item up
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                xi_values_li = x_items_values_li[item_idx]
                if 'agent' not in xi_values_li:
                    continue
                if 'pickup{}'.format(item_idx) in action_names_li:
                    instance_string = 'apickup{} *'.format(item_idx)
                    for k in range(2):
                        for item_idx2 in range(self.nr_of_items):
                            if item_idx2 in x_items_values_li.keys():
                                if item_idx == item_idx2 and k == 0:
                                    instance_string += ' -'
                                elif item_idx == item_idx2 and k == 1:
                                    instance_string += ' agent'
                                else:
                                    instance_string += ' *'
                    value_table_string = ''
                    for xi in xi_values_li:
                        if xi == 'not_here' or xi == 'agent':
                            value_table_string += ' 0.0'
                        else:
                            value_table_string += ' {}'.format(auxiliary_functions.get_pickup_reward(self.environment))
                    self.write_entry_valuetable(f, instance_string, value_table_string)
        # releasing item:
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                xi_values_li = x_items_values_li[item_idx]
                if 'agent' not in xi_values_li:
                    continue
                if 'release' in action_names_li:
                    instance_string = 'arelease *'
                    for item_idx2 in range(self.nr_of_items):
                        if item_idx2 in x_items_values_li.keys():
                            if item_idx == item_idx2:
                                instance_string += ' agent'
                            else:
                                instance_string += ' *'
                    for item_idx2 in x_items_values_li.keys():
                        instance_string += ' *'  # FOR RELEASING IN SUBTASK IT DOES NOT MATTER WHAT NEXT STATE IS
                    self.write_entry_valuetable(f, instance_string, value_table_string='-{}'.format(
                        auxiliary_functions.get_pickup_reward(self.environment)))

        # reward 0 if already in terminal state
        for s_terminal in s_terminals_li:
            instance_string = '*'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none' or (var_key != 'xa' and var_key not in x_items_values_li.keys()):
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')

        # reward 0 for illegal states
        self.write_reward_illegal_states_li(f, x_items_values_li, current_states=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def write_reward_task_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, action_names_li):
        parent_string = 'action_agent xa_0'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx)
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_1'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_task</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        # for each item:
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                xi_values = x_items_values_li[item_idx]
                if 'agent' not in xi_values or 'goal' not in xi_values or self.items[item_idx].goal_nodes_layers[
                    layer] not in xa_values_li:
                    continue
                # releasing item at goal location
                if 'release' in action_names_li:
                    instance_string = 'arelease s{}'.format(self.items[item_idx].goal_nodes_layers[layer])
                    for item_idx2 in range(self.nr_of_items):
                        if item_idx2 in x_items_values_li.keys():
                            if item_idx == item_idx2:
                                instance_string += ' agent'
                            else:
                                instance_string += ' *'
                    for item_idx2 in range(self.nr_of_items):
                        if item_idx2 in x_items_values_li.keys():
                            if item_idx == item_idx2:
                                instance_string += ' goal'
                            else:
                                instance_string += ' *'
                    self.write_entry_valuetable(f, instance_string, value_table_string='{}'.format(
                        auxiliary_functions.get_delivery_reward(self.environment)))
                # picking up item from goal location
                if 'pickup{}'.format(item_idx) in action_names_li:
                    instance_string = 'apickup{} s{}'.format(item_idx, self.items[item_idx].goal_nodes_layers[layer])
                    for item_idx2 in range(self.nr_of_items):
                        if item_idx2 in x_items_values_li.keys():
                            if item_idx == item_idx2:
                                instance_string += ' goal'
                            else:
                                instance_string += ' *'
                    for item_idx2 in range(self.nr_of_items):
                        if item_idx2 in x_items_values_li.keys():
                            if item_idx == item_idx2:
                                instance_string += ' agent'
                            else:
                                instance_string += ' *'
                    self.write_entry_valuetable(f, instance_string, value_table_string='-{}'.format(
                        auxiliary_functions.get_delivery_reward(self.environment)))

        # reward 0 if already in terminal state
        for s_terminal in s_terminals_li:
            instance_string = '*'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none' or (var_key != 'xa' and var_key not in x_items_values_li.keys()):
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')

        # reward 0 for illegal states
        self.write_reward_illegal_states_li(f, x_items_values_li, current_states=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def write_reward_illegal_states_li(self, f, x_items_values_li, current_states=True, xa_1_present=False):
        # are there multiple items that have the value agent?
        agent_counter = 0
        index_list = []
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                xi_values_li = x_items_values_li[item_idx]
                if 'agent' in xi_values_li:
                    agent_counter += 1
                    index_list.append(item_idx)
        if agent_counter > 1:
            ag1 = index_list[0]
            ag2 = index_list[1]
            for n in range(len(index_list)):
                instance_string = '* *'
                # translate ag1 ag2 into an instance
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        if item_idx == ag1 or item_idx == ag2:
                            instance_string += ' agent'
                        else:
                            instance_string += ' *'
                if xa_1_present:
                    instance_string += ' *'
                # if current state are part of parents add them to instance as *
                if current_states:
                    for k in x_items_values_li.keys():
                        instance_string += ' *'
                self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
                if ag2 == index_list[-1]:
                    ag1 = index_list[index_list.index(ag1) + 1]
                    if ag1 == index_list[-1]:
                        break
                    else:
                        ag2 = index_list[index_list.index(ag1) + 1]
                else:
                    ag2 = index_list[index_list.index(ag2) + 1]

    def write_reward_terminal_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                 a_top='none'):
        print('gets overwritten by child class')

    def write_entry_probtable(self, f, instance_string, prob_table_string):
        f.write("\t\t\t\t<Entry>\n")
        f.write("\t\t\t\t\t<Instance>{}</Instance>\n".format(instance_string))
        f.write("\t\t\t\t\t<ProbTable>{}</ProbTable>\n".format(prob_table_string))
        f.write("\t\t\t\t</Entry>\n")

    def write_entry_valuetable(self, f, instance_string, value_table_string):
        f.write("\t\t\t\t<Entry>\n")
        f.write("\t\t\t\t\t<Instance>{}</Instance>\n".format(instance_string))
        f.write("\t\t\t\t\t<ValueTable>{}</ValueTable>\n".format(value_table_string))
        f.write("\t\t\t\t</Entry>\n")




