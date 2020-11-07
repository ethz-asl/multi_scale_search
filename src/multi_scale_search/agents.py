from __future__ import division

import copy
import os
import time
import logging

import xml.etree.ElementTree as ET
import numpy as np

import config
from src.auxiliary_files.grid import Grid
from src.multi_scale_search import auxiliary_functions
from src.multi_scale_search.actions import LookAround
from src.multi_scale_search.actions import Navigate
from src.multi_scale_search.actions import Pickup
from src.multi_scale_search.actions import Release
from src.multi_scale_search.belief import Belief
from src.multi_scale_search.belief import BeliefRepresentation
from src.multi_scale_search.core import Agent
from src.multi_scale_search.core import AgentMultiScaleBasis
from src.multi_scale_search.core import NodeGraph
from src.multi_scale_search.core import POMDPProblem


class Item:
    def __init__(self, name, x, y, current_node=-1, goal_x=-1.0, goal_y=-1.0, goal_node=-1):
        self.name = name
        self.x = x
        self.y = y
        self.current_node = current_node
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_node = goal_node

    def set_goal_xy(self, goal_x, goal_y, goal_node=-1):
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_node = goal_node


class AgentFLAT(Agent):

    def __init__(self, grid=Grid(1, 1, 1, 1), pose0=(0, 0, 0), bottom_layer=1, rec_env=None, item_names=('mug'),
                 environment='None'):
        Agent.__init__(self, grid, pose0, environment)
        self.b0_threshold = 0.999
        self.nr_of_items = len(item_names)
        self.name = 'FLAT'
        self.file_name_input = config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/FLAT.pomdpx'
        self.file_name_output = config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/FLAT.policy'
        # create rectangles of nodes and nodegraph
        self.nodegraph = NodeGraph()
        self.layer = bottom_layer
        self.nr_of_layers = self.layer + 1
        if environment != 'None' and rec_env is not None:
            node_recs = config.get_recs_of_nodes(rec_env, environment=self.environment)
            # create nodegraph
            self.nodegraph = NodeGraph(nr_of_nodes=len(node_recs), node_recs=node_recs)
            config.construct_nodegraph_multiscale(self.nodegraph, layer=self.layer,
                                                  environment=self.environment)

        # initialize state
        self.s = {'xa': self.pos_to_node(pose0[0], pose0[1])}
        for item_name in item_names:
            self.s[item_name] = -1
        self.current_action_name = 'none'
        # set item-mapping
        for item_name in item_names:
            self.item_map[item_name] = len(self.item_map)
            self.items.append(Item(name=item_name, x=-1, y=-1))
        # solving related variables ####################################################################################
        node_rec_mapping = auxiliary_functions.get_nodes_recs_mapping(layer=self.layer, nr_of_layers=self.layer + 1,
                                                                      node_mapping=-1, nodegraph_lN=self.nodegraph)
        self.timeout_time = 5.0
        # initialize belief grid
        self.belief = Belief(
            total_nr_of_cells=10000 * (grid.total_width * grid.total_height) /
                              (config.grid_ref_width * config.grid_ref_height),
            recs_beliefgrids=self.nodegraph.get_all_recs_as_flat_list(), nr_of_items=len(item_names))
        self.b0 = BeliefRepresentation(self.belief, nodes_recs_mapping=node_rec_mapping)
        # initialize data structures related to the policy
        self.alpha_vectors = np.array([], dtype=float)
        self.alpha_vectors_attrib = np.array([], dtype=int)
        # ACTIONS ######################################################################################################
        # create dict of actions, key = action_name, value = action object
        self.subroutine_actions = {}
        self.action_names = []
        self.nodegraph.get_all_nav_actions(self.action_names)
        self.nr_of_nodes = len(self.nodegraph.graph.keys())
        self.xa_values = list(self.nodegraph.graph.keys())
        # navigate actions
        for action_name in self.action_names:
            self.subroutine_actions[action_name] = Navigate(
                name=action_name, file_name=action_name + '.txt', layer=self.layer, environment=self.environment,
                xa_values=self.xa_values, nodegraph=self.nodegraph, grid=self.grid)
        # look_around action
        self.subroutine_actions['look_around'] = LookAround(
            name='look_around', file_name='look_around.txt',
            layer=self.layer, environment=self.environment, xa_values=self.xa_values, nodegraph=self.nodegraph,
            node_mapping=node_rec_mapping, grid_cell_width=self.grid.cell_width, grid_cell_height=self.grid.cell_height)
        self.action_names.append('look_around')
        # release action
        self.subroutine_actions['release'] = Release(
            name='release', file_name='release.txt', layer=self.layer, environment=self.environment,
            xa_values=self.xa_values, nodegraph=self.nodegraph)
        self.action_names.append('release')
        # pickup actions
        for item_name in item_names:
            action_name = 'pickup{}'.format(self.item_map[item_name])
            self.subroutine_actions[action_name] = Pickup(
                name=action_name, file_name='pickup.txt', layer=self.layer, environment=self.environment,
                file_name_look_around='look_around.txt', xa_values=self.xa_values, nodegraph=self.nodegraph)
            self.action_names.append('pickup{}'.format(self.item_map[item_name]))
        self.subroutine_actions_history = {}
        self.log = logging.getLogger(__name__)

    def start_solving(self):
        self.initialized = True
        for key in self.item_map:
            print('item {}: {}'.format(key, self.item_map[key]))
            item = self.items[self.item_map[key]]
            print('item_goal = {}'.format(item.goal_node))
            print('belief:')
            for n in range(self.nr_of_nodes):
                b_vec = self.belief.get_aggregated_belief(rec_nr=n)
                print('b(n={})={}'.format(n, b_vec))

    def set_b0_threshold(self, b0_threshold):
        self.b0_threshold = b0_threshold

    # currently not used, might be useful for future implementation (e.g. ros)
    def add_item(self, item_type, goal_x=-1.0, goal_y=-1.0):
        if item_type not in self.item_map:
            self.s[item_type] = -1
            self.item_map[item_type] = len(self.items)
            self.items.append(Item(name=item_type, x=-1, y=-1, goal_x=goal_x, goal_y=goal_y,
                                   goal_node=self.pos_to_node(goal_x, goal_y)))
            self.nr_of_items = len(self.items)
            self.belief.add_item()
            # add pickup action for the item
            action_name = 'pickup{}'.format(self.item_map[item_type])
            self.subroutine_actions[action_name] = Pickup(
                name=action_name, file_name='pickup' + '_l{}_env{}.txt'.format(self.nr_of_layers, self.environment),
                layer=self.layer,
                file_name_look_around='look_around' + '_l{}_env{}.txt'.format(self.nr_of_layers, self.environment),
                xa_values=self.xa_values, nodegraph=self.nodegraph)
            self.action_names.append(action_name)

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
            self.items[self.item_map[item_type]].set_goal_xy(g_x, g_y, goal_node=self.pos_to_node(g_x, g_y))
        if not is_coherent:
            print('something went wrong when setting items')
            return 'error'
        else:
            return 'success'

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

    def pos_to_node(self, x, y):
        node_nr = self.nodegraph.get_node_nr(x, y)
        if node_nr == -1:
            self.log.info('error in pos_to_node mapping')
        return node_nr

    def update_carry(self, o_carry, item_x=None, item_y=None):
        for item_idx in range(self.nr_of_items):
            item_type = self.inverse_item_map(item_idx)
            if o_carry == self.items[item_idx].name:
                self.s[item_type] = 'agent'
                # set belief grid to uniform
                self.belief.set_b_tot(item_nr=self.item_map[o_carry], total_belief=0.0)
                self.belief.fill_grid(type='uniform', item_nr=self.item_map[o_carry])
            elif o_carry != self.items[item_idx].name and self.s[item_type] == 'agent':
                if item_x is not None and item_y is not None:
                    item_node = self.pos_to_node(item_x, item_y)
                else:
                    item_node = self.s['xa']
                    item_x, item_y = self.x, self.y
                if item_node == self.items[item_idx].goal_node:
                    self.s[item_type] = 'goal'
                    self.belief.set_b_tot(item_idx, 0.0)
                else:
                    self.belief.set_b_tot(item_idx, 1.0)
                    self.belief.delete_belief_spots(item_nr=item_idx)
                    self.belief.add_belief_spot(mu_x=item_x, mu_y=item_y, prob=0.99, sigma=0.01, item_nr=item_idx)
                    self.s[item_type] = -1

    def update_pose(self, o_x, o_y, o_theta):
        if not self.initialized:
            return
        # update agent configuration
        self.x, self.y, self.theta = o_x, o_y, o_theta
        self.u, self.v = self.grid.get_cell_indices_by_position(self.x, self.y)
        # update state variables
        self.s['xa'] = self.pos_to_node(self.x, self.y)

    def interpret_observations(self, o_cells):
        if not self.initialized:
            return
        # translate o_cells to item_nrs
        o_cells_nrs = []
        for o_cell in o_cells:
            if o_cell.obs_val in config.item_types:
                o_cells_nrs.append([o_cell.x, o_cell.y, self.item_map[o_cell.obs_val]])
            else:
                o_cells_nrs.append([o_cell.x, o_cell.y, o_cell.obs_val])
        if self.current_action_name == 'look_around':
            self.subroutine_actions['look_around'].update_local_grid(self.s['xa'], o_cells)
        self.belief.update_belief(observations=o_cells_nrs)

    def choose_action(self):
        # check if task is already solved
        task_finished = True
        for key in self.s.keys():
            if key == 'xa':
                continue
            if self.s[key] != 'goal':
                task_finished = False
        if task_finished:
            print('\n')
            print('***************************************************************************')
            print('*************************** task is finished ******************************')
            print('***************************************************************************')
            print('\n')
            goal_ref = 'finished'  # ('finished', (self.x, self.y, self.theta))
            return goal_ref
        self.k += 1
        a_name = self.current_action_name
        if a_name in self.subroutine_actions:
            a = self.subroutine_actions[a_name]
            # execute subroutine
            goal_ref = a.subroutine(s=self.s, item_map=self.item_map, agent_x=self.x, agent_y=self.y,
                                    agent_theta=self.theta, grid=self.grid, nodegraph=self.nodegraph,
                                    items=self.items, b0=self.b0)
        if a_name == 'none' or goal_ref == 'finished':
            t0 = time.time()
            a_name = self.compute_policy(self.nr_of_nodes, self.action_names)
            # if it is a navigation action -> set observation prob. to 0
            if a_name[0:3] == 'nav':
                n0, n1 = auxiliary_functions.nav_action_get_n0(a_name), auxiliary_functions.nav_action_get_n1(a_name)
                n_dash = -1
                if n0 == self.s['xa']:
                    n_dash = n1
                elif n1 == self.s['xa']:
                    n_dash = n0
                for nk in self.xa_values:
                    self.subroutine_actions[a_name].observation_probabilities[n_dash][nk] = 0.0
            self.current_action_name = a_name
            self.subroutine_actions_history[self.k] = a_name
            print('action = {}'.format(a_name))
            self.log.info('action = {}'.format(a_name))
            # execute subroutine
            a = self.subroutine_actions[a_name]
            goal_ref = a.subroutine(s=self.s, item_map=self.item_map, agent_x=self.x, agent_y=self.y,
                                    agent_theta=self.theta, grid=self.grid, nodegraph=self.nodegraph,
                                    items=self.items, b0=self.b0)
            t1 = time.time()
            self.computation_time_for_each_action.append(t1 - t0)
            self.computation_time += self.computation_time_for_each_action[-1]
        return goal_ref

    def compute_policy(self, nr_of_nodes, action_names):
        # create pomdpx file
        self.create_PODMPX_file(nr_of_nodes, action_names)
        # solve POMDP with SARSOP
        os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
            config.solving_precision, self.timeout_time, self.file_name_input, self.file_name_output))
        # read alpha vectors from policy file
        self.read_policy_file(self.file_name_output, nr_of_nodes)
        # compute belief for every possible state
        belief_enumerated = self.get_belief_for_every_state(nr_of_nodes)
        # now find best alpha vector for current belief
        V_best = -1000000
        a_best = 0
        sel = self.alpha_vectors_attrib[:, 0] == self.s['xa']
        for idx, vector in enumerate(self.alpha_vectors[sel]):
            V_p = np.dot(vector, belief_enumerated)
            if V_p > V_best:
                V_best = V_p
                a_best = self.alpha_vectors_attrib[sel, 1][idx]
        print('(V, a)=({}, {})'.format(V_best, a_best))
        self.log.info('(V, a)=({}, {})'.format(V_best, a_best))

        return self.action_names[a_best]

    ## FUNCTIONS RELATED TO READING THE POLICY   ########################################################################
    def read_policy_file(self, file_path, nr_of_nodes):
        tree = ET.parse(file_path)
        root = tree.getroot()
        # reshape alpha_vectors
        self.alpha_vectors = np.zeros((int(root[0].attrib['numVectors']), int(root[0].attrib['vectorLength'])))
        self.alpha_vectors_attrib = np.zeros((int(root[0].attrib['numVectors']), 2), dtype=int)
        for idx, alpha_vector in enumerate(root[0]):
            # get obs-value (i.e. xa)
            obsValue = int(alpha_vector.attrib['obsValue'])
            # get action of alpha vector
            action_nr = int(alpha_vector.attrib['action'])
            self.alpha_vectors_attrib[idx] = [obsValue, action_nr]
            values_str = alpha_vector.text
            values_str = values_str.lstrip()
            values_str = values_str.rstrip()
            values = [float(i) for i in values_str.split(' ')]
            self.alpha_vectors[idx, :] = values

    def get_belief_for_every_state(self, nr_of_nodes):
        b_of_s = np.zeros((nr_of_nodes + 2, len(self.items)))
        for n in range(nr_of_nodes):
            b_of_s[n] = self.b0.get_aggregated_belief(node_nr=n)
        b_agent, b_goal = [], []
        for i in range(self.nr_of_items):
            item_type = self.inverse_item_map(i)
            b_agent.append(float(self.s[item_type] == 'agent'))
            b_goal.append(float(self.s[item_type] == 'goal'))
        b_of_s[-2] = b_agent
        b_of_s[-1] = b_goal
        belief_enumerated = np.zeros(((nr_of_nodes + 2) ** len(self.items)))
        for s_i in range(len(belief_enumerated)):
            b = 1.0
            s_i_copy = s_i
            for item_nr in range(len(self.items) - 1, -1, -1):
                xi = int(s_i_copy / ((nr_of_nodes + 2) ** (item_nr)))
                s_i_copy = s_i_copy % ((nr_of_nodes + 2) ** (item_nr))
                b *= b_of_s[xi][len(self.items) - 1 - item_nr]
            belief_enumerated[s_i] = b
        return belief_enumerated

    ## FUNCTIONS RELATED TO CREATING POMDPX FILE #######################################################################
    def create_PODMPX_file(self, nr_of_nodes, action_names):
        # create pomdpx file
        f = open(self.file_name_input, "w+")
        # write header
        f.write("<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?>\n")
        f.write("<pomdpx version=\"1.0\" id=\"simplifiedMDP\"\n")
        f.write("\txmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n")
        f.write("\txsi:noNamespaceSchemaLocation=\"pomdpx.xsd\">\n")

        # write Description tag
        f.write("\t<Description> simplified problem for service robot. No furniture, static scene.\n")
        f.write("\t</Description>\n")
        # write Discount tag
        f.write("\t<Discount> {} </Discount>\n".format(auxiliary_functions.get_discount_factor()))
        # write Variable Tag
        self.write_variables(f, nr_of_nodes, action_names)
        # write InitialStateBelief tag
        self.write_initial_belief_function(f, nr_of_nodes)
        # write StateTransitionFunction tag
        self.write_transition_function(f, nr_of_nodes, action_names)
        # write ObsFunction Tag
        self.write_observation_function(f, nr_of_nodes, action_names)
        # write RewardFunction Tag
        self.write_reward_function(f, nr_of_nodes, action_names)
        # close pomdpx tag
        f.write("</pomdpx>")

        f.close()

    def write_variables(self, f, nr_of_nodes, action_names):
        f.write("\t<Variable>\n")
        # State Variables
        # agent position
        f.write("\t\t<StateVar vnamePrev=\"xa_0\" vnameCurr=\"xa_1\" fullyObs=\"true\">\n")
        f.write("\t\t\t<NumValues>{}</NumValues>\n".format(nr_of_nodes))
        f.write("\t\t</StateVar>\n")
        # items
        state_string = ''
        for n in range(nr_of_nodes):
            state_string += ' s{}'.format(n)
        state_string += ' agent goal'
        is_item_fully_obs = 'false'
        for item_idx in range(len(self.items)):
            f.write(
                "\t\t<StateVar vnamePrev=\"x{}_0\" vnameCurr=\"x{}_1\" fullyObs=\"".format(item_idx, item_idx) +
                is_item_fully_obs + "\">\n")
            f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(state_string))
            f.write("\t\t</StateVar>\n")

        # Observation Variables
        value_table = 'no'
        for n in range(nr_of_nodes):
            value_table += ' o{}'.format(n)
        value_table += ' agent'
        for i in range(len(self.items)):
            f.write("\t\t<ObsVar vname=\"z{}\">\n".format(i))
            f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(value_table))
            f.write("\t\t</ObsVar>\n")
        # Action Variables
        f.write("\t\t<ActionVar vname=\"action_agent\">\n")
        action_string = 'a' + action_names[0]
        for a in range(1, len(action_names)):
            action_string += ' a{}'.format(action_names[a])
        f.write("\t\t\t<ValueEnum>{}</ValueEnum>\n".format(action_string))
        f.write("\t\t</ActionVar>\n")
        # Reward Variables
        f.write("\t\t<RewardVar vname=\"reward_time\" />\n")
        f.write("\t\t<RewardVar vname=\"reward_subtask\" />\n")
        f.write("\t\t<RewardVar vname=\"reward_task\" />\n")
        f.write("\t</Variable>\n")

    def write_initial_belief_function(self, f, nr_of_nodes):
        f.write("\t<InitialStateBelief>\n")
        f.write("\t\t<CondProb>\n")
        # agent position xa
        belief_string = ''
        for node_nr in range(nr_of_nodes):
            if node_nr == self.s['xa']:
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
        b0 = copy.deepcopy(self.b0)
        for item_idx in range(len(self.items)):
            item_type = self.inverse_item_map(item_idx)
            belief_string = ''
            belief_of_nodes = {}
            set_to_one = False
            for xa in range(nr_of_nodes):
                belief_of_nodes[xa] = b0.get_aggregated_belief(node_nr=xa)[item_idx]
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
            # normalize belief
            b_sum = sum(list(belief_of_nodes.values()))
            if b_sum > 0.0:
                for key in belief_of_nodes:
                    belief_of_nodes[key] /= b_sum
            for node_nr in range(nr_of_nodes):
                prob = belief_of_nodes[node_nr]
                belief_string += ' {}'.format(prob)
            if self.s[item_type] == 'agent':
                belief_string += ' 1.0'
            else:
                belief_string += ' 0.0'
            if self.s[item_type] == 'goal':
                belief_string += ' 1.0'
            else:
                belief_string += ' 0.0'
            f.write("\t\t<CondProb>\n")
            f.write("\t\t\t<Var>x{}_0</Var>\n".format(item_idx))
            f.write("\t\t\t<Parent>null</Parent>\n")
            f.write("\t\t\t<Parameter type=\"TBL\">\n")
            self.write_entry_probtable(f, instance_string='-', prob_table_string=belief_string)
            f.write("\t\t\t</Parameter>\n")
            f.write("\t\t</CondProb>\n")
        f.write("\t</InitialStateBelief>\n")

    def write_transition_function(self, f, nr_of_nodes, action_names):
        f.write("\t<StateTransitionFunction>\n")
        self.transition_funcion_xa(f, nr_of_nodes, action_names)
        for item_idx in range(len(self.items)):
            self.transition_function_xi(item_idx, f, nr_of_nodes, action_names)
        f.write("\t</StateTransitionFunction>\n")

    def transition_funcion_xa(self, f, nr_of_nodes, action_names):
        f.write("\t\t<CondProb>\n")
        parent_string = 'action_agent xa_0'
        for i in range(len(self.items)):
            parent_string += ' x{}_0'.format(i)
        f.write("\t\t\t<Var>xa_1</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")

        for a in action_names:
            if a[0:3] == 'nav':
                i, j = auxiliary_functions.nav_action_get_n0(a), auxiliary_functions.nav_action_get_n1(a)
                prob_table_string = ''
                for node_nr in range(nr_of_nodes):
                    for node_nr2 in range(nr_of_nodes):
                        if node_nr == i and node_nr2 == j:
                            prob_table_string += ' 1.0'
                        elif node_nr == j and node_nr2 == i:
                            prob_table_string += ' 1.0'
                        elif node_nr == node_nr2 and node_nr != i and node_nr2 != i and node_nr != j and node_nr2 != j:
                            prob_table_string += ' 1.0'
                        else:
                            prob_table_string += ' 0.0'
                instance_string = 'a{}'.format(a) + ' -'
                for item_idx in range(len(self.items)):
                    instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string)
            elif a[0:6] == 'pickup':
                instance_string = 'a{}'.format(a) + ' -'
                for item_idx in range(len(self.items)):
                    instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
            elif a == 'release':
                # action = release
                instance_string = 'arelease' + ' -'
                for item_idx in range(len(self.items)):
                    instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
            elif a == 'look_around':
                instance_string = 'alook_around -'
                for item_idx in range(len(self.items)):
                    instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
        # terminal states   ############################################################################################
        instance_string = '* -'
        for item_idx in range(len(self.items)):
            instance_string += ' goal'
        instance_string += ' -'
        self.write_entry_probtable(f, instance_string, prob_table_string='identity')
        # illegal states ###############################################################################################
        if len(self.items) > 1:
            N = 0
            for item_idx in range(len(self.items)):
                N += item_idx
            ag1 = 0
            ag2 = 1
            for node_nr in range(N):
                instance_string = '* -'
                # translate ag1 ag2 into an instance
                for item_idx in range(len(self.items)):
                    if item_idx == ag1 or item_idx == ag2:
                        instance_string += ' agent'
                    else:
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
                ag2 += 1
                if ag2 == len(self.items):
                    ag1 += 1
                    ag2 = ag1 + 1
                    if ag2 == len(self.items):
                        break

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</CondProb>\n")

    def transition_function_xi(self, item_idx, f, nr_of_nodes, action_names):
        parent_string = 'action_agent xa_0'
        for item_idx2 in range(len(self.items)):
            parent_string += ' x{}_0'.format(item_idx2)
        f.write("\t\t<CondProb>\n")
        f.write("\t\t\t<Var>x{}_1</Var>\n".format(item_idx))
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")

        # action= nav(ij), xi_0 == xi_1 -> T = 1
        for a in action_names:
            if a[0:3] == 'nav':
                instance_string = 'a{}'.format(a) + ' *'
                for item_idx2 in range(len(self.items)):
                    if item_idx2 == item_idx:
                        instance_string += ' -'
                    else:
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
            elif a[0:6] == 'pickup':
                i = auxiliary_functions.pickup_get_i(a)
                if i != item_idx:
                    # as over-general rule: if xi_0 == xi_1 -> T = 1, else T=0
                    instance_string = 'a{}'.format(a) + ' *'
                    for item_idx2 in range(len(self.items)):
                        if item_idx2 == item_idx:
                            instance_string += ' -'
                        else:
                            instance_string += ' *'
                    instance_string += ' -'
                    self.write_entry_probtable(f, instance_string, prob_table_string='identity')
                # special case if i == k, P = 1.0 to transition to agent
                else:
                    self.pickup_transition_prob(item_idx, f, nr_of_nodes)
            elif a == 'release':
                # over-general rule xi_1 = xi_0
                instance_string = 'arelease' + ' *'
                for item_idx2 in range(len(self.items)):
                    if item_idx2 == item_idx:
                        instance_string += ' -'
                    else:
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
                # if xi_0 = agent -> xi_1 = xa_0
                instance_string = 'arelease -'
                for item_idx2 in range(len(self.items)):
                    if item_idx2 == item_idx:
                        instance_string += ' agent'
                    else:
                        instance_string += ' *'
                instance_string += ' -'
                prob_table = ['0.0'] * (nr_of_nodes * (nr_of_nodes + 2))
                prob_table_string = ''
                instance = [0, 0]  # instance[0] = xa_0, instance[1] = xi_1
                for idx in prob_table:
                    if instance[0] == self.items[item_idx].goal_node and instance[1] == 'goal':
                        prob_table_string += ' 1.0'
                    elif instance[0] != self.items[item_idx].goal_node and instance[0] == instance[1]:
                        prob_table_string += ' 1.0'
                    else:
                        prob_table_string += ' 0.0'
                    # increase instance by 1
                    for k in range(len(instance) - 1, -1, -1):
                        if instance[k] == nr_of_nodes - 1:
                            instance[k] = 'agent'
                            break
                        elif instance[k] == 'agent':
                            instance[k] = 'goal'
                            break
                        elif instance[k] == 'goal':
                            instance[k] = 0
                        else:
                            instance[k] += 1
                            break
                self.write_entry_probtable(f, instance_string, prob_table_string)
            elif a == 'look_around':
                instance_string = 'alook_around *'
                for item_idx2 in range(len(self.items)):
                    if item_idx2 == item_idx:
                        instance_string += ' -'
                    else:
                        instance_string += ' *'
                instance_string += ' -'
                self.write_entry_probtable(f, instance_string, prob_table_string='identity')
        # terminal states   ########################################################################################
        instance_string = '* *'
        for item_idx2 in range(len(self.items)):
            instance_string += ' goal'
        instance_string += ' goal'
        self.write_entry_probtable(f, instance_string, prob_table_string='1.0')
        # illegal states:
        if len(self.items) > 1:
            N = 0
            for item_idx2 in range(len(self.items)):
                N += item_idx2
            ag1 = 0
            ag2 = 1
            for n in range(N):
                instance_string = '* *'
                # translate ag1 ag2 into an instance
                for item_idx2 in range(len(self.items)):
                    if item_idx2 == ag1 or item_idx2 == ag2:
                        instance_string += ' agent'
                    elif item_idx2 == item_idx:
                        instance_string += ' -'
                    else:
                        instance_string += ' *'
                instance_string += ' -'
                prob_table_string = 'identity'
                if instance_string.count('-') == 1:
                    prob_table_string = ''
                    for node_nr in range(nr_of_nodes):
                        prob_table_string += '0.0 '
                    prob_table_string += '1.0 0.0'
                self.write_entry_probtable(f, instance_string, prob_table_string)
                ag2 += 1
                if ag2 == len(self.items):
                    ag1 += 1
                    ag2 = ag1 + 1
                    if ag2 == len(self.items):
                        break

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</CondProb>\n")

    def pickup_transition_prob(self, item_i, f, nr_of_nodes):
        # rule 1: xa=-, x0 = -, ..., xi = -, xi' = '    ################################################################
        instance_string = 'apickup{} -'.format(item_i)
        for item_idx in range(len(self.items)):
            if item_idx == item_i:
                instance_string += ' -'
            else:
                instance_string += ' *'
        instance_string += ' -'
        # prob table
        prob_table_string = ''
        instance = [0, 0, 0]
        nr_of_states = nr_of_nodes * (nr_of_nodes + 2) ** 2
        for state_nr in range(nr_of_states):
            if instance[0] == instance[1] and instance[2] == 'agent':
                prob_table_string += ' 1.0'
            elif instance[0] == self.items[item_i].goal_node and instance[1] == 'goal' and instance[2] == 'agent':
                prob_table_string += ' 1.0'
            elif instance[0] == self.items[item_i].goal_node and instance[1] == 'goal' and instance[1] == instance[2]:
                prob_table_string += ' 0.0'
            elif instance[0] != instance[1] and instance[1] == instance[2]:
                prob_table_string += ' 1.0'
            else:
                prob_table_string += ' 0.0'
            # increase instance by 1
            for k in range(len(instance) - 1, -1, -1):
                if instance[k] == nr_of_nodes - 1:
                    instance[k] = 'agent'
                    break
                elif instance[k] == 'agent':
                    instance[k] = 'goal'
                    break
                elif instance[k] == 'goal':
                    instance[k] = 0
                else:
                    instance[k] += 1
                    break
        self.write_entry_probtable(f, instance_string, prob_table_string)
        # rule 2: some other xj is already 'agent' #####################################################################
        for item_idx in range(len(self.items)):
            if item_idx == item_i:
                continue
            instance_string = 'apickup{} *'.format(item_i)
            for item_idx2 in range(len(self.items)):
                if item_idx == item_idx2:
                    instance_string += ' agent'
                elif item_idx2 == item_i:
                    instance_string += ' -'
                else:
                    instance_string += ' *'
            instance_string += ' -'
            self.write_entry_probtable(f, instance_string, prob_table_string='identity')
        # rule 3: terminal state #######################################################################################
        instance_string = 'apickup{} *'.format(item_i)
        for item_idx in range(len(self.items)):
            instance_string += ' goal'
        instance_string += ' -'
        prob_table_string = ''
        for n in range(nr_of_nodes):
            prob_table_string += ' 0.0'
        prob_table_string += ' 0.0 1.0'
        self.write_entry_probtable(f, instance_string, prob_table_string)

    def write_observation_function(self, f, nr_of_nodes, action_names):
        f.write("\t<ObsFunction>\n")
        # loop over observation variables
        for item_idx in range(len(self.items)):
            f.write("\t\t<CondProb>\n")
            parent_string = 'action_agent xa_1 x{}_1'.format(item_idx)
            f.write("\t\t\t<Var>z{}</Var>\n".format(item_idx))
            f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
            f.write("\t\t\t<Parameter type = \"TBL\">\n")
            # for each action a rule is written
            for a in action_names:
                if a[0:3] == 'nav':
                    i, j = auxiliary_functions.nav_action_get_n0(a), auxiliary_functions.nav_action_get_n1(a)
                    # general rule: prob. of observing zi = no is 1 for all states
                    instance_string = 'a{} * * -'.format(a)
                    prob_table_string = '1.0'  # prob. 1 of observing "no"
                    for node_nr in range(nr_of_nodes):
                        prob_table_string += ' 0.0'
                    prob_table_string += ' 0.0'
                    self.write_entry_probtable(f, instance_string, prob_table_string)
                    # overwrite rule with exception: if xa=i, xi= j, zi = j -> P = 0.6, if xa=i, xi=i, zi=i -> P=0.4
                    self.obs_function_nav(a, i, f, nr_of_nodes)
                    self.obs_function_nav(a, j, f, nr_of_nodes)
                elif a[0:6] == 'pickup':
                    # things that are observed during a pickup action are not considered, there is a
                    # look_around action for observing the node
                    instance_string = 'a{} * * no'.format(a)
                    prob_table_string = '1.0'
                    self.write_entry_probtable(f, instance_string, prob_table_string)
                elif a == 'release':
                    instance_string = 'a{} * * no'.format(a)
                    prob_table_string = '1.0'
                    self.write_entry_probtable(f, instance_string, prob_table_string)
                elif a == 'look_around':
                    instance_string = 'a{} - - -'.format(a)
                    prob_table_string = ''
                    for idx_xa in range(nr_of_nodes):
                        for idx_xk in range(2 + nr_of_nodes):
                            for idx_zk in range(2 + nr_of_nodes):
                                if idx_xa == idx_xk and idx_xa + 1 == idx_zk:
                                    value = '{}'.format(self.subroutine_actions[a].observation_probabilities[idx_xa])
                                elif idx_xa == idx_xk and idx_zk == 0:
                                    value = '{}'.format(
                                        1 - self.subroutine_actions[a].observation_probabilities[idx_xa])
                                elif idx_xk == nr_of_nodes and idx_zk == 1 + nr_of_nodes:
                                    value = '1.0'
                                elif idx_zk == 0 and idx_xk != nr_of_nodes:
                                    value = '1.0'
                                else:
                                    value = '0.0'
                                prob_table_string += ' {}'.format(value)
                    self.write_entry_probtable(f, instance_string, prob_table_string)
            # for all actions: if xk = agent, zk is 'agent' with P=1.0
            instance_string = '* * agent -'
            prob_table_string = '0.0'
            for n in range(nr_of_nodes):
                prob_table_string += ' 0.0'
            prob_table_string += ' 1.0'
            self.write_entry_probtable(f, instance_string, prob_table_string)

            # close variable
            f.write("\t\t\t</Parameter>\n")
            f.write("\t\t</CondProb>\n")
        f.write("\t</ObsFunction>\n")

    def obs_function_nav(self, a, i, f, nr_of_nodes):
        instance_string = 'a{} s{} - -'.format(a, i)
        prob_table_string = ''
        # loop over all item positions:
        for idx_xk in range(2 + nr_of_nodes):
            # loop over all observation values
            for idx_zk in range(2 + nr_of_nodes):
                if a in self.subroutine_actions_history.values():
                    if idx_xk != nr_of_nodes and idx_zk == 0:
                        value = '1.0'
                    elif idx_xk == nr_of_nodes and idx_zk == nr_of_nodes + 1:
                        value = '1.0'
                    else:
                        value = '0.0'
                elif idx_xk + 1 == idx_zk and idx_xk != nr_of_nodes:
                    value = '{}'.format(self.subroutine_actions[a].observation_probabilities[i][idx_xk])
                elif idx_zk == 0 and idx_xk != nr_of_nodes and idx_xk != nr_of_nodes + 1:
                    value = '{}'.format(1 - self.subroutine_actions[a].observation_probabilities[i][idx_xk])
                elif idx_xk == nr_of_nodes and idx_zk == nr_of_nodes + 1:
                    value = '1.0'
                elif idx_xk == nr_of_nodes + 1 and idx_zk == 0:
                    value = '1.0'
                else:
                    value = '0.0'
                prob_table_string += ' ' + value
        self.write_entry_probtable(f, instance_string, prob_table_string)

    def write_reward_function(self, f, nr_of_nodes, action_names):
        f.write("\t<RewardFunction>\n")
        self.write_reward_time(f, nr_of_nodes, action_names)
        self.write_reward_sub_task(f, nr_of_nodes)
        self.write_reward_task(f)
        f.write("\t</RewardFunction>\n")

    def write_reward_time(self, f, nr_of_nodes, action_names):
        parent_string = 'action_agent xa_0'
        for i in range(len(self.items)):
            parent_string += ' x{}_0'.format(i)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_time</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        for a in action_names:
            if a[0:3] == 'nav':
                instance_string = 'a{} -'.format(a)
                for j in range(len(self.items)):
                    instance_string += ' *'
                value_table_string = ''
                for xa in self.xa_values:
                    value_table_string += ' -{}'.format(self.subroutine_actions[a].t_expected[xa])
                self.write_entry_valuetable(f, instance_string, value_table_string)
            elif a == 'look_around':
                instance_string = 'a{} -'.format(a)
                for item_idx in range(len(self.items)):
                    instance_string += ' -'
                value_table_string = ''
                for state_nr in range(nr_of_nodes * (nr_of_nodes + 2) ** (len(self.items))):
                    state = self.deenumerate_state(s_i=state_nr, nr_of_nodes=nr_of_nodes)
                    is_item_present = False
                    for s_i in state[1:]:
                        if state[0] == s_i:
                            value_table_string += ' -{}'.format(self.subroutine_actions[a].t_expected[1][state[0]])
                            is_item_present = True
                            break
                    if not is_item_present:
                        value_table_string += ' -{}'.format(self.subroutine_actions[a].t_expected[0][state[0]])
                self.write_entry_valuetable(f, instance_string, value_table_string)
            elif a[0:6] == 'pickup':
                i = int(a[6])
                instance_string = 'a{} -'.format(a)
                for item_idx in range(len(self.items)):
                    if item_idx == i:
                        instance_string += ' -'
                    else:
                        instance_string += ' *'
                value_table_string = ''
                for xa in self.xa_values:
                    for xi in range(2 + nr_of_nodes):
                        if xa == xi or (xa == self.items[i].goal_node and xi == nr_of_nodes + 1):
                            value_table_string += ' -{}'.format(self.subroutine_actions[a].t_expected[1][xa])
                        elif xi == nr_of_nodes:
                            value_table_string += ' -1.0'
                        else:
                            value_table_string += ' -{}'.format(self.subroutine_actions[a].t_expected[0][xa])
                self.write_entry_valuetable(f, instance_string, value_table_string)
            elif a == 'release':
                # general rule: negative reward of release_time for release action
                instance_string = 'arelease *'
                for item_idx in range(len(self.items)):
                    instance_string += ' *'
                self.write_entry_valuetable(f, instance_string, value_table_string='-{}'.format(
                    config.robot_release_time))
                # release when a variable is actually 'agent'
                for item_idx in range(len(self.items)):
                    instance_string = 'arelease -'
                    for item_idx2 in range(len(self.items)):
                        if item_idx2 == item_idx:
                            instance_string += ' agent'
                        else:
                            instance_string += ' *'
                    value_table_string = ''
                    for xa in self.xa_values:
                        if xa == self.items[item_idx].goal_node:
                            value_table_string += ' -{}'.format(self.subroutine_actions[a].t_expected[xa])
                        else:
                            value_table_string += ' -{}'.format(config.robot_release_time)
                    self.write_entry_valuetable(f, instance_string, value_table_string)
        # reward 0 if all items in goal-location (terminal state)
        instance_string = '* *'  # for all actions and all xa
        for i in range(len(self.items)):
            instance_string += ' goal'
        self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
        # reward 0 for illegal states
        self.write_reward_illegal_state(f, current_states=False)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def write_reward_sub_task(self, f, nr_of_nodes):
        parent_string = 'action_agent xa_0'
        for item_idx in range(len(self.items)):
            parent_string += ' x{}_0'.format(item_idx)
        for item_idx in range(len(self.items)):
            parent_string += ' x{}_1'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_subtask</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")

        # for each item, picking item up
        for item_idx in range(len(self.items)):
            instance_string = 'apickup{} *'.format(item_idx)  # for pickupi action and xa in relation to xi
            for k in range(2):
                for item_idx2 in range(len(self.items)):
                    if item_idx == item_idx2 and k == 0:
                        instance_string += ' -'
                    elif item_idx == item_idx2 and k == 1:
                        instance_string += ' agent'
                    else:
                        instance_string += ' *'
            value_table_string = ''
            for node_idx in range(nr_of_nodes):
                value_table_string += ' 20'
            value_table_string += ' 0.0 20'
            self.write_entry_valuetable(f, instance_string, value_table_string)

        # releasing item:
        for item_idx in range(len(self.items)):
            instance_string = 'arelease *'  # for release action and all xa
            for item_idx2 in range(len(self.items)):
                if item_idx == item_idx2:
                    instance_string += ' agent'
                else:
                    instance_string += ' *'
            for item_idx2 in range(len(self.items)):
                instance_string += ' *'  # FOR RELEASING IN SUBTASK IT DOES NOT MATTER WHAT NEXT STATE IS
            self.write_entry_valuetable(f, instance_string, value_table_string='-20.0')

        # reward 0 if all items in goal-location (terminal state)
        instance_string = '* *'  # for all actions and all xa
        for item_idx in range(len(self.items)):
            instance_string += ' goal'
        for item_idx in range(len(self.items)):
            instance_string += ' *'
        self.write_entry_valuetable(f, instance_string, value_table_string='0.0')

        # reward 0 for illegal states
        self.write_reward_illegal_state(f, current_states=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def write_reward_task(self, f):
        parent_string = 'action_agent xa_0'
        for item_idx in range(len(self.items)):
            parent_string += ' x{}_0'.format(item_idx)
        for item_idx in range(len(self.items)):
            parent_string += ' x{}_1'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_task</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        # for each item:
        for item_idx in range(len(self.items)):
            # releasing item at goal location
            instance_string = 'arelease s{}'.format(self.items[item_idx].goal_node)
            for item_idx2 in range(len(self.items)):
                if item_idx == item_idx2:
                    instance_string += ' agent'
                else:
                    instance_string += ' *'
            for item_idx2 in range(len(self.items)):
                if item_idx == item_idx2:
                    instance_string += ' goal'
                else:
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='100.0')
            # picking up item from goal location
            instance_string = 'apickup{} s{}'.format(item_idx, self.items[item_idx].goal_node)
            for item_idx2 in range(len(self.items)):
                if item_idx == item_idx2:
                    instance_string += ' goal'
                else:
                    instance_string += ' *'
            for item_idx2 in range(len(self.items)):
                if item_idx == item_idx2:
                    instance_string += ' agent'
                else:
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='-100.0')
        # reward 0 if all items in goal-location (terminal state)
        instance_string = '* *'  # for all actions and all xa
        for item_idx in range(len(self.items)):
            instance_string += ' goal'
        for item_idx in range(len(self.items)):
            instance_string += ' *'
        self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
        # reward 0 for illegal states
        self.write_reward_illegal_state(f, current_states=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def write_reward_illegal_state(self, f, current_states=True):
        if len(self.items) > 1:
            N = 0
            for k in range(len(self.items)):
                N += k
            ag1 = 0
            ag2 = 1
            for n in range(N):
                instance_string = '* *'
                # translate ag1 ag2 into an instance
                for item_idx in range(len(self.items)):
                    if item_idx == ag1 or item_idx == ag2:
                        instance_string += ' agent'
                    else:
                        instance_string += ' *'
                # if current state are part of parents add them to instance as *
                if current_states:
                    for item_idx in range(len(self.items)):
                        instance_string += ' *'
                self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
                ag2 += 1
                if ag2 == len(self.items):
                    ag1 += 1
                    ag2 = ag1 + 1
                    if ag2 == len(self.items):
                        break

    # careful: the state consists only of numbers, variable value 'agent' corresponds to the number nr_of_nodes
    def deenumerate_state(self, s_i, nr_of_nodes):
        s_i_copy = s_i
        s = [0] * (1 + len(self.items))
        for i in range(len(s)):
            s[i] = int(s_i_copy / ((2 + nr_of_nodes) ** (len(self.items) - i)))
            s_i_copy = s_i_copy % ((2 + nr_of_nodes) ** (len(self.items) - i))
        return np.array(s)

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


# Method M1 corresponds to Action-Propagation (AP)
class AgentMultiScaleM1(AgentMultiScaleBasis):
    def __init__(self, grid=Grid(1, 1, 1, 1), conf0=(0, 0, 0), nr_of_layers=2, node_mapping=None, rec_env=None,
                 item_names=('mug'), environment='None'):
        AgentMultiScaleBasis.__init__(self, grid, conf0, nr_of_layers, node_mapping, rec_env, item_names, environment)
        self.name = 'MultiScaleM1'
        self.file_names_input = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM1_input_l{}.pomdpx'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.file_names_output = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM1_output_l{}.policy'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.log = logging.getLogger(__name__)

    def get_state_action_observation_sets(self, s_li, s_top, layer, a_name_top, nr_of_items):
        # create set S^l2, S_t^l2 and A^l2
        xa_values_li, x_items_values_li = [], {}  # x_items_value_l2 = {item_nr: [item_values]}
        s_terminals_li = []  # [{'xa':.., 0:.., ..}, {..}, ...]
        action_names_li = []
        z_values_li = {}  # key = item_nr, value = list of values
        if layer == 0:
            return self.get_state_action_observation_sets_l0(s_li, nr_of_items)
        elif a_name_top[0:3] == 'nav':
            n0, n1 = auxiliary_functions.nav_action_get_n0(a_name_top), auxiliary_functions.nav_action_get_n1(
                a_name_top)
            ni, nj = -1, -1
            if s_top['xa'] == n0:
                ni = n0
                nj = n1
            elif s_top['xa'] == n1:
                nj = n0
                ni = n1
            else:
                print('action a_l{}={} is not available in s_l{}={}'.format(layer - 1, a_name_top, layer - 1, s_top))
                self.log.info('action a_l{}={} is not available in s_l{}={}'.format(layer - 1, a_name_top, layer - 1, s_top))
            xa_values_li = self.get_subnodes(node=ni, layer=layer - 1)
            # get all neighbour nodes
            neighbours = []
            for n in xa_values_li:
                neighbours += self.nodegraphs_layers[layer].get_neighbour_nodes(node_nr=n)
            # delete duplicates
            neighbours = list(set(neighbours))
            # add all neighbours that are in n^l1_j to xa_values_l2 and xa_terminals_l2
            for n in neighbours:
                if n in self.get_subnodes(node=nj, layer=layer - 1):
                    xa_values_li.append(n)
                    s_terminal = {'xa': [n]}
                    for item_idx in range(nr_of_items):
                        s_terminal[item_idx] = ['none']
                    s_terminals_li.append(s_terminal)
            action_names_li = self.nodegraphs_layers[layer].get_nav_actions_for_nodes(
                nodes=self.get_subnodes(node=ni, layer=layer - 1), nodes_restriction=xa_values_li)
        elif a_name_top[0:6] == 'pickup':
            item_idx = int(a_name_top[6])
            item_type = self.inverse_item_map(item_idx)
            xa_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
            xi_values_li, zi_values_li = [], []
            if s_top[item_type] == 'agent':
                xi_values_li = ['agent']
            elif s_top[item_type] == 'goal':
                if self.items[item_idx].goal_nodes_layers[layer - 1] == s_top['xa']:
                    xi_values_li = ['agent', 'goal']
                else:
                    xi_values_li = ['goal']
            else:
                xi_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['not_here', 'agent']
                zi_values_li = ['no'] + self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['agent']
            x_items_values_li[item_idx] = xi_values_li
            z_values_li[item_idx] = zi_values_li
            s_terminal = {'xa': ['*']}
            for item_idx2 in range(nr_of_items):
                if item_idx2 == item_idx:
                    s_terminal[item_idx2] = ['agent']
                else:
                    s_terminal[item_idx2] = ['none']
            s_terminals_li.append(s_terminal)
            action_names_li = self.nodegraphs_layers[layer].get_nav_actions_for_nodes(nodes=xa_values_li,
                                                                                      nodes_restriction=xa_values_li)
            if layer == self.nr_of_layers - 1:
                action_names_li.append('look_around')
            action_names_li.append('pickup{}'.format(item_idx))
        elif a_name_top == 'release':
            for item_idx2 in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx2)
                if s_top[item_type] == 'agent':
                    item_idx = item_idx2
            xa_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
            terminal_var_list = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
            x_items_values_li[item_idx] = self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['agent']
            if self.items[item_idx].goal_nodes_layers[layer] in terminal_var_list:
                x_items_values_li[item_idx] += ['goal']
                terminal_var_list += ['goal']
                if self.items[item_idx].goal_nodes_layers[layer] in x_items_values_li:
                    x_items_values_li[item_idx].remove(self.items[item_idx].goal_nodes_layers[layer])
                    terminal_var_list.remove(self.items[item_idx].goal_nodes_layers[layer])
            z_values_li[item_idx] = ['no'] + self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['agent']
            for x_item_value in terminal_var_list:
                s_terminal = {'xa': self.get_subnodes(node=s_top['xa'], layer=layer - 1)}
                for item_idx2 in range(nr_of_items):
                    if item_idx2 == item_idx:
                        s_terminal[item_idx2] = [x_item_value]
                    else:
                        s_terminal[item_idx2] = ['none']
                s_terminals_li.append(s_terminal)
            action_names_li = self.nodegraphs_layers[layer].get_nav_actions_for_nodes(nodes=xa_values_li,
                                                                                      nodes_restriction=xa_values_li)
            action_names_li.append('release')
        # sort xa_values
        xa_values_li.sort()
        return xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li

    def write_reward_terminal_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                 a_top='none'):
        if self.solve_terminal_states_problems == 'no':
            parent_string = 'action_agent xa_0'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_0'.format(item_idx)
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_1'.format(item_idx)
            f.write("\t\t<Func>\n")
            f.write("\t\t\t<Var>reward_terminal</Var>\n")
            f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
            f.write("\t\t\t<Parameter type = \"TBL\">\n")
            instance_string = '* *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
        else:
            # create header ############################################################################################
            parent_string = 'action_agent xa_0'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_0'.format(item_idx)
            parent_string += ' xa_1'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_1'.format(item_idx)
            f.write("\t\t<Func>\n")
            f.write("\t\t\t<Var>reward_terminal</Var>\n")
            f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
            f.write("\t\t\t<Parameter type = \"TBL\">\n")
            # write instance for transfering to the terminal states ####################################################
            for s_terminal in s_terminals_li:
                instance_string = '*'  # for any action
                instance_string += ' *'  # for any starting xa
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                key_list = ['xa'] + list(range(0, self.nr_of_items))
                for var_key in key_list:
                    if var_key in s_terminal.keys():
                        var_t = s_terminal[var_key]
                        if var_t[0] == 'none' or (var_key != 'xa' and var_key not in x_items_values_li.keys()):
                            continue
                        elif var_t[0] == '*' or len(var_t) > 1:
                            instance_string += ' -'
                        elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                            instance_string += ' {}'.format(var_t[0])
                        else:
                            instance_string += ' s{}'.format(var_t[0])
                # calculate reward for transfering to terminal states ##################################################
                value_table_string = self.get_terminal_reward_li(layer, s_terminal)
                self.write_entry_valuetable(f, instance_string, value_table_string)
            # 0 reward for starting and staying in terminal state ######################################################
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
                instance_string += ' *'
                for item_idx in range(self.nr_of_items):
                    if item_idx in x_items_values_li.keys():
                        instance_string += ' *'
                self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
            # reward 0 for illegal states
            self.write_reward_illegal_states_li(f, x_items_values_li, current_states=True, xa_1_present=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def get_terminal_reward_li(self, layer, s_terminal_li):
        value_table_string = ''
        # create belief_enumerated for terminal state in next Problem
        s_terminal = copy.copy(s_terminal_li)
        for key in s_terminal.keys():
            s_terminal[key] = s_terminal_li[key][0]
        belief_enumerated_next = self.get_belief_for_every_state(layer, s_terminal, self.b0_layers[layer],
                                                                 self.POMDPs[layer]['next'].x_items_values,
                                                                 self.POMDPs[layer]['next'].z_values)
        # read Value from POMDP next
        V_best = -100000
        sel = self.get_alpha_vectors_selection(s_terminal, self.POMDPs[layer]['next'].xa_values,
                                               self.POMDPs[layer]['next'].x_items_values,
                                               self.POMDPs[layer]['next'].z_values,
                                               self.POMDPs[layer]['next'].alpha_vectors_attrib)
        for idx, vector in enumerate(self.POMDPs[layer]['next'].alpha_vectors[sel]):
            V_p = np.dot(vector, belief_enumerated_next)
            if V_p > V_best:
                V_best = V_p
        value_table_string += ' {}'.format(V_best)
        return value_table_string


# Method M2 corresponds to Action-Value-Propagation (AVP)
class AgentMultiScaleM2(AgentMultiScaleBasis):
    def __init__(self, grid=Grid(1, 1, 1, 1), conf0=(0, 0, 0), nr_of_layers=2, node_mapping=None, rec_env=None,
                 item_names=('mug'), environment='None'):
        AgentMultiScaleBasis.__init__(self, grid, conf0, nr_of_layers, node_mapping, rec_env, item_names, environment)
        self.name = 'MultiScaleM2'
        self.file_names_input = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM2_input_l{}.pomdpx'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.file_names_output = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM2_output_l{}.policy'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.special_terminal_states = {}
        self.log = logging.getLogger(__name__)

    def compute_policy(self, layer, a_name_top):
        xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li = \
            self.get_state_action_observation_sets(self.s_layers[layer], self.s_layers[layer - 1], layer, a_name_top,
                                                   len(self.items))
        if layer > 0:
            POMDP_key = self.var_lower_to_var_top(layer, self.s_layers[layer]['xa'])
        else:
            POMDP_key = 'all'
        if len(self.POMDPs) >= layer + 1:
            self.POMDPs[layer][POMDP_key].set_variable_sets(
                xa_values_li, x_items_values_li, s_terminals_li, z_values_li, action_names_li)
        else:
            self.POMDPs.append({POMDP_key: POMDPProblem(xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                                        action_names_li)})
        # sort terminal states according to xa_top value
        terminal_state_mapping = {}
        if layer > 0:
            for s_terminal in s_terminals_li:
                if s_terminal['xa'][0] != '*' and s_terminal['xa'][0] not in \
                        self.get_subnodes(node=self.s_layers[layer - 1]['xa'], layer=layer - 1):
                    xa_top_next = self.var_lower_to_var_top(layer_lower=layer, var_lower=s_terminal['xa'][0])
                else:
                    continue
                if xa_top_next not in terminal_state_mapping.keys():
                    terminal_state_mapping[xa_top_next] = [s_terminal]
                else:
                    terminal_state_mapping[xa_top_next].append(s_terminal)
            # delete all terminal states in terminal_state_mapping that don't require solving another POMDP problem
            for xa_top_next in list(terminal_state_mapping.keys()):
                if len(terminal_state_mapping[xa_top_next]) == 1:
                    del terminal_state_mapping[xa_top_next]
        self.special_terminal_states = terminal_state_mapping
        top_layer = layer - 1
        for xa_top_next, terminal_states in self.special_terminal_states.items():
            # get next action move of layer above
            s_top_next = self.s_layers[top_layer].copy()
            s_top_next['xa'] = xa_top_next
            # get a starting state for this layer in s_top_next
            s_copy = self.s_layers[layer].copy()
            s_copy['xa'] = self.get_subnodes(node=s_top_next['xa'], layer=top_layer)[0]  # terminal_states[0]['xa']
            if layer > 1:
                POMDP_top_key = self.var_lower_to_var_top(layer_lower=top_layer, var_lower=xa_top_next)
            else:
                POMDP_top_key = 'all'
            a_name_top_next = self.read_policy(s_top_next, top_layer, self.POMDPs[top_layer][POMDP_top_key].xa_values,
                                               self.POMDPs[top_layer][POMDP_top_key].x_items_values,
                                               self.POMDPs[top_layer][POMDP_top_key].z_values,
                                               self.POMDPs[top_layer][POMDP_top_key].action_names, POMDP_top_key,
                                               alpha_already_computed=True)
            xa_values_li_next, x_items_values_li_next, s_terminals_li_next, action_names_li_next, z_values_li_next = \
                self.get_state_action_observation_sets(s_copy, s_top_next, layer, a_name_top=a_name_top_next,
                                                       nr_of_items=len(self.items))
            self.POMDPs[layer][xa_top_next] = POMDPProblem(xa_values_li_next, x_items_values_li_next,
                                                           s_terminals_li_next,
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
            self.POMDPs[layer][xa_top_next].set_alpha_vectors(alpha_vectors, alpha_vectors_attrib)
        #     self.solve_terminal_states_problems = True

        self.solve_terminal_states_problems = len(terminal_state_mapping) > 0
        # # self.solving_style = 'M3'
        self.create_POMDPX_file(self.s_layers[layer], layer, xa_values_li, x_items_values_li, s_terminals_li,
                                action_names_li, z_values_li)
        # solve POMDP with SARSOP
        os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
            config.solving_precision, self.timeout_time,
            self.file_names_input[layer], self.file_names_output[layer]))
        a_name_li = self.read_policy(self.s_layers[layer], layer, xa_values_li, x_items_values_li, z_values_li,
                                     action_names_li, POMDP_key=POMDP_key)
        # a_name_li = self.read_policy(self.s_layers[layer], layer, xa_values_li, x_items_values_li, z_values_li,
        #                              action_names_li, POMDP_key='current')
        self.solve_terminal_states_problems = False
        return a_name_li

    def get_state_action_observation_sets(self, s_li, s_top, layer, a_name_top, nr_of_items):
        # create set S^l2, S_t^l2 and A^l2
        xa_values_li, x_items_values_li = [], {}  # x_items_value_li = {item_nr: [item_values]}
        s_terminals_li = []  # [[xa_t0, x0_t0, .., xn_t0], [xa_t1, ...], ...]
        action_names_li = []
        z_values_li = {}  # key = item_nr, value = list of values
        if layer == 0:
            return self.get_state_action_observation_sets_l0(s_li, nr_of_items)
        elif a_name_top[0:3] == 'nav':
            n0, n1 = auxiliary_functions.nav_action_get_n0(a_name_top), auxiliary_functions.nav_action_get_n1(
                a_name_top)
            ni, nj = -1, -1
            if s_top['xa'] == n0:
                ni = n0
                nj = n1
            elif s_top['xa'] == n1:
                nj = n0
                ni = n1
            else:
                print('action a_l{}={} is not available in s_l{}={}'.format(layer - 1, a_name_top, layer - 1, s_top))
                self.log.info('action a_l{}={} is not available in s_l{}={}'.format(layer - 1, a_name_top, layer - 1, s_top))
            xa_values_li = self.get_subnodes(node=ni, layer=layer - 1)
            # get all neighbour nodes
            neighbours = []
            for n in xa_values_li:
                neighbours += self.nodegraphs_layers[layer].get_neighbour_nodes(node_nr=n)
            # delete duplicates
            neighbours = list(set(neighbours))
            # add all neighbours that are in n^l1_j to xa_values_l2 and xa_terminals_l2
            for n in neighbours:
                if n in self.get_subnodes(node=nj, layer=layer - 1):
                    xa_values_li.append(n)
            for n in neighbours:
                if n in self.get_subnodes(node=nj, layer=layer - 1):
                    s_terminal = {'xa': [n]}
                    for item_idx in range(nr_of_items):
                        item_type = self.inverse_item_map(item_idx)
                        if self.s_layers[layer][item_type] != -1:
                            s_terminal[item_idx] = [self.s_layers[layer][item_type]]
                        else:
                            s_terminal[item_idx] = copy.copy(xa_values_li) + ['not_here']
                    s_terminals_li.append(s_terminal)
            for item_idx in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx)
                if s_li[item_type] == 'agent':
                    xj_values_li = ['agent']
                elif s_li[item_type] == 'goal':
                    xj_values_li = ['goal']
                else:
                    xj_values_li = copy.copy(xa_values_li) + ['not_here']
                    zj_values_li = ['no'] + copy.copy(xa_values_li)
                    z_values_li[item_idx] = zj_values_li
                x_items_values_li[item_idx] = xj_values_li
            action_names_li = self.nodegraphs_layers[layer].get_nav_actions_for_nodes(
                nodes=self.get_subnodes(node=ni, layer=layer - 1), nodes_restriction=xa_values_li)
            if layer == self.nr_of_layers - 1:
                action_names_li.append('look_around')
        elif a_name_top[0:6] == 'pickup':
            item_idx = int(a_name_top[6])
            xa_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
            xj_values_li = []
            for item_idx2 in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx2)
                if s_top[item_type] == 'agent':
                    xj_values_li = ['agent']
                elif s_top[item_type] == 'goal':
                    xj_values_li = ['goal']
                else:
                    xj_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['not_here']
                    z_values_li[item_idx2] = ['no'] + self.get_subnodes(node=s_top['xa'], layer=layer - 1)
                    if item_idx == item_idx2:
                        xj_values_li.append('agent')
                        z_values_li[item_idx2].append('agent')
                x_items_values_li[item_idx2] = xj_values_li
            s_terminal = {'xa': self.get_subnodes(node=s_top['xa'], layer=layer - 1)}
            for item_idx2 in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx2)
                if item_idx2 == item_idx:
                    s_terminal[item_idx2] = ['agent']
                elif s_top[item_type] == 'goal':
                    s_terminal[item_idx2] = ['goal']
                else:
                    s_terminal[item_idx2] = ['*']
            s_terminals_li.append(s_terminal)
            action_names_li = self.nodegraphs_layers[layer].get_nav_actions_for_nodes(nodes=xa_values_li,
                                                                                      nodes_restriction=xa_values_li)
            if layer == self.nr_of_layers - 1:
                action_names_li.append('look_around')
            action_names_li.append('pickup{}'.format(item_idx))
        elif a_name_top == 'release':
            for item_idx2 in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx2)
                if s_li[item_type] == 'agent':
                    item_idx = item_idx2
            xa_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
            for item_idx2 in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx2)
                if item_idx2 == item_idx:
                    xj_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1) + \
                                   ['agent']
                    if self.items[item_idx].goal_nodes_layers[layer] in xj_values_li:
                        xj_values_li.append('goal')
                        if layer == self.nr_of_layers - 1:
                            xj_values_li.remove(self.items[item_idx].goal_nodes_layers[layer])
                elif s_li[item_type] == 'goal':
                    xj_values_li = ['goal']
                else:
                    xj_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['not_here']
                    z_values_li[item_idx2] = ['no'] + self.get_subnodes(node=s_top['xa'], layer=layer - 1)
                x_items_values_li[item_idx2] = xj_values_li
            terminal_values = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
            if self.items[item_idx].goal_nodes_layers[layer] in terminal_values:
                terminal_values.append('goal')
            for x_item_value in terminal_values:
                if layer == self.nr_of_layers - 1 and x_item_value == self.items[item_idx].goal_nodes_layers[layer]:
                    continue
                s_terminal = {'xa': ['*']}
                for item_idx2 in range(nr_of_items):
                    if item_idx2 == item_idx:
                        s_terminal[item_idx2] = [x_item_value]
                    else:
                        s_terminal[item_idx2] = ['*']
                s_terminals_li.append(s_terminal)
            action_names_li = self.nodegraphs_layers[layer].get_nav_actions_for_nodes(nodes=xa_values_li,
                                                                                      nodes_restriction=xa_values_li)
            if layer == self.nr_of_layers - 1:
                action_names_li.append('look_around')
            action_names_li.append('release')
        # sort xa_values
        xa_values_li.sort()
        return xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li

    def write_reward_terminal_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                 a_top='none'):
        if layer == 0:
            parent_string = 'action_agent xa_0'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_0'.format(item_idx)
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_1'.format(item_idx)
            f.write("\t\t<Func>\n")
            f.write("\t\t\t<Var>reward_terminal</Var>\n")
            f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
            f.write("\t\t\t<Parameter type = \"TBL\">\n")
            instance_string = '* *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
            f.write("\t\t\t</Parameter>\n")
            f.write("\t\t</Func>\n")
            return
        # create header ################################################################################################
        parent_string = 'action_agent xa_0'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx)
        parent_string += ' xa_1'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_1'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_terminal</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        # write instance for transfering to the terminal states ########################################################
        for s_terminal in s_terminals_li:
            instance_string = '*'  # for any action
            instance_string += ' *'  # for any starting xa
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none':
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' -'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            # calculate reward for transfering to terminal states ######################################################
            value_table_string = self.get_terminal_reward_li(layer, xa_values_li, x_items_values_li, s_terminal)
            self.write_entry_valuetable(f, instance_string, value_table_string)
        # 0 reward for starting and staying in terminal state ######################################################
        for s_terminal in s_terminals_li:
            instance_string = '*'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none':
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            instance_string += ' *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
        # reward 0 for illegal states
        self.write_reward_illegal_states_li(f, x_items_values_li, current_states=True, xa_1_present=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def get_terminal_reward_li(self, layer, xa_values_li, x_items_values_li, s_terminal_li):
        value_table_string = ''
        # STEP 1: get all terminal state variations
        var_values_li = {}
        key_list = ['xa'] + list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in s_terminal_li.keys():
                var_val = s_terminal_li[var_name]
                if var_val[0] == '*' or len(var_val) > 1:
                    if var_name == 'xa':
                        var_values_li[var_name] = xa_values_li
                    else:
                        var_values_li[var_name] = x_items_values_li[var_name]
                elif var_val[0] == 'none':
                    continue
                else:
                    var_values_li[var_name] = [var_val[0]]
        nr_of_terminal_states = np.prod([len(var_values_li[key]) for key in var_values_li.keys()])
        # STEP 2: loop over all terminal states
        s_li = {}
        key_list = ['xa'] + list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in var_values_li.keys():
                var_values = var_values_li[var_name]
                s_li[var_name] = var_values[0]
        for state_nr in range(nr_of_terminal_states):
            # if impossible terminal state -> - 10000 reward
            impossible_state = False
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_name in key_list:
                if var_name in s_li.keys():
                    var_value = s_li[var_name]
                    if var_value not in s_terminal_li[var_name] and s_terminal_li[var_name][0] != 'none' and \
                            s_terminal_li[var_name][0] != '*':
                        value_table_string += ' -1000000'
                        impossible_state = True
                        break
            if not impossible_state:
                use_next_POMDP = False
                for terminal_states in self.special_terminal_states.values():
                    for terminal_state in terminal_states:
                        if s_li['xa'] == terminal_state['xa'][0]:
                            use_next_POMDP = True
                if use_next_POMDP and self.solve_terminal_states_problems:
                    # check if terminal belief is also a terminal belief in Problem next
                    next_layer = layer
                    s_next = copy.copy(s_li)
                    for higher_layer in range(layer, -1, -1):
                        next_layer = higher_layer
                        if next_layer == 0:
                            POMDP_key = 'all'
                            break
                        elif next_layer > 0:
                            POMDP_key = self.var_lower_to_var_top(next_layer, s_next['xa'])
                        for s_terminal_next in self.POMDPs[next_layer][POMDP_key].s_terminals:
                            different_states = False
                            key_list = ['xa'] + list(range(0, self.nr_of_items))
                            for var_name in key_list:
                                if var_name in s_terminal_next.keys():
                                    var_val = s_terminal_next[var_name]
                                    if var_val[0] == '*' or len(var_val) > 1:
                                        continue
                                    if var_val[0] != s_next[var_name]:
                                        different_states = True
                                        break
                            if not different_states:
                                break
                        if different_states:
                            break
                        for var_name, var_val in list(s_next.items()):
                            s_next[var_name] = self.var_lower_to_var_top(layer_lower=next_layer, var_lower=var_val)
                        next_layer -= 1
                    # create belief_enumerated
                    if next_layer == layer:
                        belief_enumerated_next = self.get_terminal_belief_for_every_state_next(
                            next_layer, s_next, self.b0_layers[next_layer],
                            self.POMDPs[next_layer][POMDP_key].x_items_values,
                            self.POMDPs[next_layer][POMDP_key].z_values, x_items_values_li)
                    else:
                        belief_enumerated_next = self.get_terminal_belief_for_every_state_top(
                            next_layer, layer, self.b0_layers[next_layer], s_next,
                            self.POMDPs[next_layer][POMDP_key].x_items_values,
                            self.POMDPs[next_layer][POMDP_key].z_values, b0_li=self.b0_layers[layer],
                            x_items_values_li=x_items_values_li)

                    # read Values from the terminal state POMDP
                    V_best = -100000
                    sel = self.get_alpha_vectors_selection(s_next, self.POMDPs[next_layer][POMDP_key].xa_values,
                                                           self.POMDPs[next_layer][POMDP_key].x_items_values,
                                                           self.POMDPs[next_layer][POMDP_key].z_values,
                                                           self.POMDPs[next_layer][POMDP_key].alpha_vectors_attrib)
                    for idx, vector in enumerate(self.POMDPs[next_layer][POMDP_key].alpha_vectors[sel]):
                        V_p = np.dot(vector, belief_enumerated_next)
                        if V_p > V_best:
                            V_best = V_p
                    value_table_string += ' {}'.format(
                        V_best + 5000)  # quick fix: to avoid discounting problems, add a constant value to all values
                else:
                    # STEP 2a: convert s_li to s_top
                    s_top = copy.copy(s_li)
                    top_layer = layer - 1
                    for higher_layer in range(layer - 1, -1, -1):
                        top_layer = higher_layer
                        # set s_top
                        for var_name, var_val in list(s_top.items()):
                            s_top[var_name] = self.var_lower_to_var_top(layer_lower=higher_layer + 1, var_lower=var_val)
                        # get POMDP key
                        if higher_layer == 0:
                            POMDP_key = 'all'
                            break
                        POMDP_key = self.var_lower_to_var_top(higher_layer, s_top['xa'])
                        # if the key does not exist, check if POMDP_key is a terminal state of another POMDP problem
                        if POMDP_key not in self.POMDPs[higher_layer].keys():
                            for key in self.POMDPs[higher_layer].keys():
                                for s_terminal in self.POMDPs[higher_layer][key].s_terminals:
                                    if POMDP_key == self.var_lower_to_var_top(higher_layer, s_terminal['xa'][0]):
                                        POMDP_key = key
                                        break
                        # check if s_top is not a terminal state in the layer above
                        for s_terminal_top in self.POMDPs[higher_layer][POMDP_key].s_terminals:
                            different_states = False
                            for var_name, var_val in s_terminal_top.items():
                                if var_val[0] == '*' or len(var_val) > 1:
                                    continue
                                if var_val[0] != s_top[var_name]:
                                    different_states = True
                                    break
                            if not different_states:
                                break
                        if different_states:
                            break
                    # STEP 2b: create b_top from s_top
                    belief_top_enumerated = self.get_terminal_belief_for_every_state_top(
                        top_layer, layer, self.b0_layers[top_layer], s_top,
                        self.POMDPs[top_layer][POMDP_key].x_items_values,
                        self.POMDPs[top_layer][POMDP_key].z_values, self.b0_layers[layer], x_items_values_li)
                    # STEP 2c: read Value of top layer with b_top and set as reward
                    V_best = -1000000
                    sel = self.get_alpha_vectors_selection(s_top, self.POMDPs[top_layer][POMDP_key].xa_values,
                                                           self.POMDPs[top_layer][POMDP_key].x_items_values,
                                                           self.POMDPs[top_layer][POMDP_key].z_values,
                                                           self.POMDPs[top_layer][POMDP_key].alpha_vectors_attrib)
                    for idx, vector in enumerate(self.POMDPs[top_layer][POMDP_key].alpha_vectors[sel]):
                        V_p = np.dot(vector, belief_top_enumerated)
                        if V_p > V_best:
                            V_best = V_p
                    value_table_string += ' {}'.format(V_best + 5000)
            # change s_li to next state
            keys_all = ['xa'] + list(range(0, self.nr_of_items))
            keys = []
            for key in keys_all:
                if key in s_li.keys():
                    keys.append(key)
            keys.reverse()
            for key in keys:
                if s_li[key] == var_values_li[key][-1]:
                    s_li[key] = var_values_li[key][0]
                else:
                    next_idx = var_values_li[key].index(s_li[key]) + 1
                    s_li[key] = var_values_li[key][next_idx]
                    break

        return value_table_string

    def get_terminal_belief_for_every_state_top(self, top_layer, layer_li, b0_top, s_top, x_items_values_top,
                                                z_values_top, b0_li, x_items_values_li):
        s_top_copy = copy.copy(s_top)
        # remove variables from s_top_copy that are not in z_values_top
        for var_name in s_top.keys():
            if var_name not in z_values_top.keys():
                del s_top_copy[var_name]
        # add variables to s_top that are not in s_li
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys() and item_idx in z_values_top.keys() and item_idx not in s_top_copy.keys():
                s_top_copy[item_idx] = -1
        b_of_s_top = []
        key_list = list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in s_top_copy.keys():
                b_of_var = []
                for idx, var_val in enumerate(x_items_values_top[var_name]):
                    if s_top_copy[var_name] != -1 and s_top_copy[var_name] != 'not_here' and s_top_copy[
                        var_name] == var_val:
                        b_of_var.append(1.0)
                    elif s_top_copy[var_name] != -1 and s_top_copy[var_name] != 'not_here' and s_top_copy[
                        var_name] != var_val:
                        b_of_var.append(0.0)
                    elif s_top_copy[var_name] == -1 or s_top_copy[var_name] == 'not_here' and (
                            var_val == 'agent' or var_val == 'goal'):
                        b_of_var.append(0.0)
                    elif s_top_copy[var_name] == -1 or s_top_copy[var_name] == 'not_here':
                        if var_val == 'not_here':
                            b_top_not_here = 0
                            for xi_top in self.xa_values_all_layers[top_layer]:
                                if xi_top not in x_items_values_top[var_name]:
                                    b_xi_top = b0_top.get_aggregated_belief(node_nr=xi_top)[var_name]
                                    # WARNING: if below if condition is changed, need to change it in
                                    # "write_initial_belief_function_li", "get_belief_for_every_state",
                                    # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                                    if top_layer >= 0:
                                        if b_xi_top > self.b0_threshold:
                                            b_xi_top = 1.0
                                        elif b_xi_top < 1 - self.b0_threshold:
                                            b_xi_top = 0.0
                                    b_top_not_here += b_xi_top
                            b_of_var.append(b_top_not_here)
                        else:
                            b_of_xi = 0.0
                            # xi_values_li_of_xi_top = self.get_subnodes(node=var_val, layer=top_layer)
                            if top_layer + 1 != layer_li:
                                debug = True
                            xi_values_li_of_xi_top = self.get_subnodes_over_multiple_layers(
                                top_node=var_val, top_layer=top_layer, bottom_layer=layer_li)
                            for xi_li in x_items_values_li[var_name]:
                                if xi_li in xi_values_li_of_xi_top:
                                    b_xi_li = b0_li.get_aggregated_belief(node_nr=xi_li)[var_name]
                                    # WARNING: if below if condition is changed, need to change it in
                                    # "write_initial_belief_function_li", "get_belief_for_every_state",
                                    # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                                    if top_layer >= 0:
                                        if b_xi_li > self.b0_threshold:
                                            b_xi_li = 1.0
                                        elif b_xi_li < 1 - self.b0_threshold:
                                            b_xi_li = 0.0
                                    b_of_xi += b_xi_li
                            b_of_var.append(b0_top.get_aggregated_belief(node_nr=var_val)[var_name] - b_of_xi)
                # normalize row
                row_sum = sum(b_of_var)
                if row_sum > 0:
                    for idx, var_val in enumerate(x_items_values_top[var_name]):
                        b_of_var[idx] /= row_sum
                b_of_s_top.append(b_of_var)
        nr_of_item_states = 1
        for var_name in s_top_copy.keys():
            nr_of_item_states *= len(x_items_values_top[var_name])
        belief_top_enumerated = np.zeros(nr_of_item_states)

        var_indices = []
        key_list = list(range(0, self.nr_of_items))
        for var_key in key_list:
            if var_key in s_top_copy.keys():
                var_indices.append(var_key)
        for s_j in range(len(belief_top_enumerated)):
            b = 1.0
            s_j_copy = s_j
            for item_idx in range(len(var_indices)):
                nr_of_xj_values = 1
                for idx in range(len(var_indices) - 1, item_idx, -1):
                    nr_of_xj_values *= len(x_items_values_top[var_indices[idx]])
                xj_top_idx = int(s_j_copy / nr_of_xj_values)
                s_j_copy = s_j_copy % nr_of_xj_values
                b *= b_of_s_top[item_idx][xj_top_idx]
            belief_top_enumerated[s_j] = b
        return belief_top_enumerated

    def get_terminal_belief_for_every_state_next(self, layer, s_li, b0_next, x_items_values_li_next, z_values_li_next,
                                                 x_items_values_li_curr):
        s_li_copy = copy.copy(s_li)
        # remove variables from s_li_copy that are not in z_values_li_next
        for var_name in s_li.keys():
            if var_name not in z_values_li_next.keys():
                del s_li_copy[var_name]
        b_of_s_next = []
        key_list = list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in s_li_copy.keys():
                b_of_var = []
                for idx, var_val in enumerate(x_items_values_li_next[var_name]):
                    if s_li_copy[var_name] != 'not_here' and s_li_copy[var_name] == var_val:
                        b_of_var.append(1.0)
                    elif s_li_copy[var_name] != 'not_here' and var_val != 'not_here' and s_li_copy[var_name] != var_val:
                        b_of_var.append(0.0)
                    elif s_li_copy[var_name] != 'not_here' and var_val == 'not_here':
                        if s_li_copy[var_name] not in x_items_values_li_next[var_name]:
                            b_of_var.append(1.0)
                        else:
                            b_of_var.append(0.0)
                    elif s_li_copy[var_name] == 'not_here':
                        if var_val != 'not_here' and (
                                var_val in x_items_values_li_curr[var_name] or var_val == 'agent' or var_val == 'goal'):
                            b_of_var.append(0.0)
                        elif var_val != 'not_here':
                            b_next = b0_next.get_aggregated_belief(node_nr=var_val)[var_name]
                            # WARNING: if below if condition is changed, need to change it in
                            # "write_initial_belief_function_li", "get_belief_for_every_state",
                            # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                            if layer >= 0:
                                if b_next > self.b0_threshold:
                                    b_next = 1.0
                                elif b_next < 1 - self.b0_threshold:
                                    b_next = 0.0
                            b_of_var.append(b_next)
                        else:
                            b_of_xj = 0
                            for xa_lj in self.xa_values_all_layers[layer]:
                                if xa_lj not in x_items_values_li_curr[var_name] and xa_lj not in \
                                        x_items_values_li_next[var_name]:
                                    b_next = b0_next.get_aggregated_belief(node_nr=xa_lj)[var_name]
                                    # WARNING: if below if condition is changed, need to change it in
                                    # "write_initial_belief_function_li", "get_belief_for_every_state",
                                    # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                                    if layer >= 0:
                                        if b_next > self.b0_threshold:
                                            b_next = 1.0
                                        elif b_next < 1 - self.b0_threshold:
                                            b_next = 0.0
                                    b_of_xj += b_next
                            b_of_var.append(b_of_xj)
                # normalize row
                row_sum = sum(b_of_var)
                if row_sum == 0:
                    debug = True
                else:
                    for idx, var_val in enumerate(x_items_values_li_next[var_name]):
                        b_of_var[idx] /= row_sum
                b_of_s_next.append(b_of_var)

        nr_of_item_states = 1
        for var_name in s_li_copy.keys():
            nr_of_item_states *= len(x_items_values_li_next[var_name])
        belief_next_enumerated = np.zeros(nr_of_item_states)
        var_indices = []
        key_list = list(range(0, self.nr_of_items))
        for var_key in key_list:
            if var_key in s_li_copy.keys():
                var_indices.append(var_key)

        for s_j in range(len(belief_next_enumerated)):
            b = 1.0
            s_j_copy = s_j
            for item_idx in range(len(var_indices)):
                nr_or_xj_values = 1
                for idx in range(len(var_indices) - 1, item_idx, -1):
                    nr_or_xj_values *= len(x_items_values_li_next[var_indices[idx]])
                xj_next_idx = int(s_j_copy / nr_or_xj_values)
                s_j_copy = s_j_copy % nr_or_xj_values
                b *= b_of_s_next[item_idx][xj_next_idx]
            belief_next_enumerated[s_j] = b
        return belief_next_enumerated


# Method M3 corresponds to Value-Propagation (VP).
# Note that M3 can have stability issues which result in navigating between to same set of nodes for ever.
class AgentMultiScaleM3(AgentMultiScaleBasis):
    def __init__(self, grid=Grid(1, 1, 1, 1), conf0=(0, 0, 0), nr_of_layers=2, node_mapping=None, rec_env=None,
                 item_names=('mug'), environment='None'):
        AgentMultiScaleBasis.__init__(self, grid, conf0, nr_of_layers, node_mapping, rec_env, item_names, environment)
        self.name = 'MultiScaleM3'
        self.file_names_input = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM3_input_l{}.pomdpx'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.file_names_output = [
            config.BASE_FOLDER_SIM + self.environment + '_environment/input_output_files/MSM3_output_l{}.policy'.format(
                1 + layer) for layer in range(self.nr_of_layers)]
        self.special_terminal_states = {}
        self.log = logging.getLogger(__name__)

    def compute_policy(self, layer, a_name_top):
        xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li = \
            self.get_state_action_observation_sets(self.s_layers[layer], self.s_layers[layer - 1], layer, a_name_top,
                                                   len(self.items))
        if layer > 0:
            POMDP_key = self.var_lower_to_var_top(layer, self.s_layers[layer]['xa'])
        else:
            POMDP_key = 'all'
        if len(self.POMDPs) >= layer + 1:
            self.POMDPs[layer][POMDP_key].set_variable_sets(
                xa_values_li, x_items_values_li, s_terminals_li, z_values_li, action_names_li)
        else:
            self.POMDPs.append({POMDP_key: POMDPProblem(xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                                        action_names_li)})
        # check if a terminal state is also a terminal state of POMDP one layer below, if so, need to solve next problem as well
        top_layer = layer - 1
        # sort terminal states according to xa_top value
        terminal_state_mapping = {}
        if layer > 0:
            for s_terminal in s_terminals_li:
                if s_terminal['xa'][0] != '*' and s_terminal['xa'][0] not in \
                        self.get_subnodes(node=self.s_layers[layer - 1]['xa'], layer=layer - 1):
                    xa_top_next = self.var_lower_to_var_top(layer_lower=layer, var_lower=s_terminal['xa'][0])
                else:
                    continue
                if xa_top_next not in terminal_state_mapping.keys():
                    terminal_state_mapping[xa_top_next] = [s_terminal]
                else:
                    terminal_state_mapping[xa_top_next].append(s_terminal)
            # delete all terminal states in terminal_state_mapping that don't require solving another POMDP problem
            for xa_top_next in list(terminal_state_mapping.keys()):
                if len(terminal_state_mapping[xa_top_next]) == 1:
                    # get terminal states for layer below (if it exists)
                    if layer < self.nr_of_layers - 1:
                        below_layer = layer + 1
                        _, _, s_terminals_below, _, _ = \
                            self.get_state_action_observation_sets(self.s_layers[below_layer], self.s_layers[layer],
                                                                   below_layer, a_name_top='none',
                                                                   nr_of_items=len(self.items))
                        xa_terminals_below = [s_terminals_below[i]['xa'] for i in range(len(s_terminals_below)) if
                                              s_terminals_below[i]['xa'][0] not in self.get_subnodes(
                                                  node=self.s_layers[layer]['xa'], layer=layer)]

                        xa_terminals_below_mapped = [self.var_lower_to_var_top(below_layer, xa_terminals_below[i][0])
                                                     for i in range(len(xa_terminals_below))]

                        xa_terminals_li = [s_terminals_li[i]['xa'][0] for i in range(len(s_terminals_li)) if
                                           s_terminals_li[i]['xa'][0] in self.get_subnodes(node=xa_top_next,
                                                                                           layer=top_layer)]

                        delete_entry = True
                        for xa_li in xa_terminals_li:
                            if xa_li in xa_terminals_below_mapped:
                                delete_entry = False
                        if delete_entry:
                            del terminal_state_mapping[xa_top_next]
                    else:
                        del terminal_state_mapping[xa_top_next]
        self.special_terminal_states = copy.copy(terminal_state_mapping)
        for xa_top_next, terminal_states in terminal_state_mapping.items():
            # get next action move of layer above
            s_top_next = copy.copy(self.s_layers[top_layer])
            s_top_next['xa'] = xa_top_next
            # get a starting state for this layer in s_top_next
            s_copy = copy.copy(self.s_layers[layer])
            s_copy['xa'] = self.get_subnodes(node=s_top_next['xa'], layer=top_layer)[0]  # terminal_states[0]['xa']
            if layer > 1:
                POMDP_top_key = self.var_lower_to_var_top(layer_lower=top_layer, var_lower=xa_top_next)
            else:
                POMDP_top_key = 'all'
            a_name_top_next = self.read_policy(s_top_next, top_layer, self.POMDPs[top_layer][POMDP_top_key].xa_values,
                                               self.POMDPs[top_layer][POMDP_top_key].x_items_values,
                                               self.POMDPs[top_layer][POMDP_top_key].z_values,
                                               self.POMDPs[top_layer][POMDP_top_key].action_names, POMDP_top_key,
                                               alpha_already_computed=True)
            xa_values_li_next, x_items_values_li_next, s_terminals_li_next, action_names_li_next, z_values_li_next = \
                self.get_state_action_observation_sets(s_copy, s_top_next, layer, a_name_top=a_name_top_next,
                                                       nr_of_items=len(self.items))
            self.POMDPs[layer][xa_top_next] = POMDPProblem(xa_values_li_next, x_items_values_li_next,
                                                           s_terminals_li_next,
                                                           z_values_li_next, action_names_li_next)
            # create POMDPX file and solve with SARSOP
            self.create_POMDPX_file(s_copy, layer, xa_values_li_next, x_items_values_li_next, s_terminals_li_next,
                                    action_names_li_next, z_values_li_next, a_name_top_next)
            os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
                config.solving_precision, self.timeout_time,
                self.file_names_input[layer], self.file_names_output[layer]))
            # get alpha vectors which are used for solving the actual problem
            alpha_vectors, alpha_vectors_attrib = self.read_alpha_vectors(self.file_names_output[layer],
                                                                          xa_values_li_next, x_items_values_li_next)
            self.POMDPs[layer][xa_top_next].set_alpha_vectors(alpha_vectors, alpha_vectors_attrib)
            self.solve_terminal_states_problems = True

        self.create_POMDPX_file(self.s_layers[layer], layer, xa_values_li, x_items_values_li, s_terminals_li,
                                action_names_li, z_values_li, a_name_top)
        # solve POMDP with SARSOP
        os.system(config.SARSOP_SRC_FOLDER + "./pomdpsol --precision {} --timeout {} {} --output {}".format(
            config.solving_precision, self.timeout_time,
            self.file_names_input[layer], self.file_names_output[layer]))
        a_name_li = self.read_policy(self.s_layers[layer], layer, xa_values_li, x_items_values_li, z_values_li,
                                     action_names_li, POMDP_key=POMDP_key)
        self.solve_terminal_states_problems = False
        return a_name_li

    def get_state_action_observation_sets(self, s_li, s_top, layer, a_name_top, nr_of_items):
        xa_values_li, x_items_values_li = [], {}  # x_items_value_l2 = {item_nr: [item_values]}
        s_terminals_li = []  # [[xa_t0, x0_t0, .., xn_t0], [xa_t1, ...], ...]
        action_names_li = []
        z_values_li = {}  # key = item_nr, value = list of values
        if layer == 0:
            return self.get_state_action_observation_sets_l0(s_li, nr_of_items)
        # get xa values ################################################################################################
        xa_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
        # get all neighbouring terminal nodes
        neighbours = []
        for n in xa_values_li:
            neighbours += self.nodegraphs_layers[layer].get_neighbour_nodes(node_nr=n)
        # delete duplicates
        neighbours = list(set(neighbours))
        # add all neighbour-nodes to xa_values and to terminal nodes
        for n in neighbours:
            if n not in xa_values_li:
                xa_values_li.append(n)
        # sort xa_values
        xa_values_li.sort()
        # add all terminal states where xa=n
        for n in neighbours:
            if n not in self.get_subnodes(node=s_top['xa'], layer=layer - 1):
                s_terminal = {'xa': [n]}
                for item_idx in range(nr_of_items):
                    item_type = self.inverse_item_map(item_idx)
                    if self.s_layers[layer][item_type] != -1:
                        s_terminal[item_idx] = [self.s_layers[layer][item_type]]
                    else:
                        s_terminal[item_idx] = copy.copy(xa_values_li) + ['not_here']
                s_terminals_li.append(s_terminal)
        # get all actions ##############################################################################################
        # add navigation actions to actions
        action_names_li += self.nodegraphs_layers[layer].get_nav_actions_for_nodes(
            nodes=self.get_subnodes(node=s_top['xa'], layer=layer - 1), nodes_restriction=xa_values_li)
        if layer == self.nr_of_layers - 1:
            action_names_li.append('look_around')
        if 'agent' not in s_top.values():
            for item_idx in range(nr_of_items):
                item_type = self.inverse_item_map(item_idx)
                if s_li[item_type] != 'goal':
                    action_names_li.append('pickup{}'.format(item_idx))
        else:
            action_names_li.append('release')
        # get all item values ##########################################################################################
        for item_idx in range(nr_of_items):
            item_type = self.inverse_item_map(item_idx)
            zj_values_li = []
            if s_li[item_type] == 'agent':
                xj_values_li = self.get_subnodes(node=s_top['xa'], layer=layer - 1) + ['agent']
                terminal_var_list = self.get_subnodes(node=s_top['xa'], layer=layer - 1)
                if self.items[item_idx].goal_nodes_layers[layer] in self.get_subnodes(node=s_top['xa'],
                                                                                      layer=layer - 1):
                    xj_values_li += ['goal']
                    terminal_var_list += ['goal']
                    if layer == self.nr_of_layers - 1:
                        if self.items[item_idx].goal_nodes_layers[layer] in terminal_var_list:
                            terminal_var_list.remove(self.items[item_idx].goal_nodes_layers[layer])
                            xj_values_li.remove(self.items[item_idx].goal_nodes_layers[layer])
                # zj_values_li = ['no'] + xa_values_li.copy() + ['agent']
                # add terminal states for releasing item in any node
                for n in terminal_var_list:
                    if n != 'goal':
                        s_terminal = {'xa': [n]}
                    else:
                        s_terminal = {'xa': [self.items[item_idx].goal_nodes_layers[layer]]}
                    for item_idx2 in range(nr_of_items):
                        item_type2 = self.inverse_item_map(item_idx2)
                        if item_idx2 == item_idx:
                            s_terminal[item_idx2] = [n]
                        else:
                            if self.s_layers[layer][item_type2] != -1:
                                s_terminal[item_idx2] = [self.s_layers[layer][item_type2]]
                            else:
                                s_terminal[item_idx2] = ['*']
                    s_terminals_li.append(s_terminal)
            elif s_li[item_type] == 'goal':
                xj_values_li = ['goal']
                # zj_values_li = ['no']
            else:
                xj_values_li = copy.copy(xa_values_li) + ['not_here']
                zj_values_li = ['no'] + copy.copy(xa_values_li)
                # add terminal state for picking item up
                if 'agent' not in self.s_layers[0].values():
                    xj_values_li += ['agent']
                    zj_values_li += ['agent']
                    s_terminal = {'xa': self.get_subnodes(node=s_top['xa'], layer=layer - 1)}
                    for item_idx2 in range(nr_of_items):
                        item_type2 = self.inverse_item_map(item_idx2)
                        if item_idx2 == item_idx:
                            s_terminal[item_idx2] = ['agent']
                        else:
                            if self.s_layers[layer][item_type2] != -1:
                                s_terminal[item_idx2] = [self.s_layers[layer][item_type2]]
                            else:
                                s_terminal[item_idx2] = ['*']
                    s_terminals_li.append(s_terminal)
            x_items_values_li[item_idx] = xj_values_li
            if len(zj_values_li) > 0:
                z_values_li[item_idx] = zj_values_li

        return xa_values_li, x_items_values_li, s_terminals_li, action_names_li, z_values_li

    def write_reward_terminal_li(self, f, layer, xa_values_li, x_items_values_li, s_terminals_li, z_values_li,
                                 a_top='none'):
        if layer == 0:
            parent_string = 'action_agent xa_0'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_0'.format(item_idx)
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    parent_string += ' x{}_1'.format(item_idx)
            f.write("\t\t<Func>\n")
            f.write("\t\t\t<Var>reward_terminal</Var>\n")
            f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
            f.write("\t\t\t<Parameter type = \"TBL\">\n")
            instance_string = '* *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
            f.write("\t\t\t</Parameter>\n")
            f.write("\t\t</Func>\n")
            return

        # create header ################################################################################################
        parent_string = 'action_agent xa_0'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_0'.format(item_idx)
        parent_string += ' xa_1'
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_li.keys():
                parent_string += ' x{}_1'.format(item_idx)
        f.write("\t\t<Func>\n")
        f.write("\t\t\t<Var>reward_terminal</Var>\n")
        f.write("\t\t\t<Parent>{}</Parent>\n".format(parent_string))
        f.write("\t\t\t<Parameter type = \"TBL\">\n")
        # write instance for transfering to the terminal states ########################################################
        for s_terminal in s_terminals_li:
            instance_string = '*'  # for any action
            instance_string += ' *'  # for any starting xa
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none':
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' -'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            # calculate reward for transfering to terminal states ######################################################
            value_table_string = self.get_terminal_reward_li(layer, xa_values_li, x_items_values_li, s_terminal, a_top)
            self.write_entry_valuetable(f, instance_string, value_table_string)
        # 0 reward for starting and staying in terminal state ######################################################
        for s_terminal in s_terminals_li:
            instance_string = '*'
            for var_key in key_list:
                if var_key in s_terminal.keys():
                    var_t = s_terminal[var_key]
                    if var_t[0] == 'none':
                        continue
                    elif var_t[0] == '*' or len(var_t) > 1:
                        instance_string += ' *'
                    elif var_t[0] == 'agent' or var_t[0] == 'goal' or var_t[0] == 'not_here':
                        instance_string += ' {}'.format(var_t[0])
                    else:
                        instance_string += ' s{}'.format(var_t[0])
            instance_string += ' *'
            for item_idx in range(self.nr_of_items):
                if item_idx in x_items_values_li.keys():
                    instance_string += ' *'
            self.write_entry_valuetable(f, instance_string, value_table_string='0.0')
        # reward 0 for illegal states
        self.write_reward_illegal_states_li(f, x_items_values_li, current_states=True, xa_1_present=True)

        f.write("\t\t\t</Parameter>\n")
        f.write("\t\t</Func>\n")

    def get_terminal_reward_li(self, layer, xa_values_li, x_items_values_li, s_terminal_li, a_top):
        # STEP 0: precompute reward penalty if s_terminal_li is not the terminal state suggested by layer above
        # get terminal state from layer above
        V_penalty = 0.0
        s_terminal_suggested = {}
        if a_top[0:3] == 'nav':
            n0 = auxiliary_functions.nav_action_get_n0(a_top)
            n1 = auxiliary_functions.nav_action_get_n1(a_top)
            if self.s_layers[layer - 1]['xa'] == n0:
                s_terminal_suggested['xa'] = n1
            else:
                s_terminal_suggested['xa'] = n0
            if len(s_terminal_li['xa']) > 0:
                if s_terminal_suggested['xa'] == self.var_lower_to_var_top(layer_lower=layer,
                                                                           var_lower=s_terminal_li['xa'][0]):
                    V_penalty = 0.0
                else:
                    V_penalty = config.get_penalty_reward(self.environment, layer)
        elif a_top[0:6] == 'pickup':
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_name in key_list:
                if var_name in s_terminal_li.keys():
                    var_val = s_terminal_li[var_name]
                    if var_val[0] == 'agent':
                        V_penalty = 0.0
                        break
                    V_penalty = config.get_penalty_reward(self.environment, layer)
        elif a_top[0:6] == 'release':
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_name in key_list:
                if var_name in s_terminal_li.keys():
                    var_val = s_terminal_li[var_name]
                    if var_val[0] != 'agent' and len(var_val[0]) == 1:
                        V_penalty = 0.0
                        break
                    V_penalty = config.get_penalty_reward(self.environment, layer)

        # STEP 1: get all terminal state variations
        value_table_string = ''
        var_values_li = {}
        key_list = ['xa'] + list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in s_terminal_li.keys():
                var_val = s_terminal_li[var_name]
                if var_val[0] == '*' or len(var_val) > 1:
                    if var_name == 'xa':
                        var_values_li[var_name] = xa_values_li
                    else:
                        var_values_li[var_name] = x_items_values_li[var_name]
                elif var_val[0] == 'none':
                    continue
                else:
                    var_values_li[var_name] = [var_val[0]]
        nr_of_terminal_states = np.prod([len(var_values_li[key]) for key in var_values_li.keys()])
        # STEP 2: loop over all terminal states
        s_li = {}
        key_list = ['xa'] + list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in var_values_li.keys():
                var_values = var_values_li[var_name]
                s_li[var_name] = var_values[0]
        for state_nr in range(nr_of_terminal_states):
            # if impossible terminal state -> - 10000 reward
            impossible_state = False
            key_list = ['xa'] + list(range(0, self.nr_of_items))
            for var_name in key_list:
                if var_name in s_li.keys():
                    var_value = s_li[var_name]
                    if var_value not in s_terminal_li[var_name] and s_terminal_li[var_name][0] != 'none' and \
                            s_terminal_li[var_name][0] != '*':
                        value_table_string += ' 0'
                        impossible_state = True
                        break
            if not impossible_state and list(s_li.values()).count('agent') > 1:
                value_table_string += ' 0'
                impossible_state = True
            if not impossible_state:
                use_next_POMDP = False
                for terminal_states in self.special_terminal_states.values():
                    for terminal_state in terminal_states:
                        if s_li['xa'] == terminal_state['xa'][0]:
                            use_next_POMDP = True
                if use_next_POMDP and self.solve_terminal_states_problems:
                    # check if terminal belief is also a terminal belief in Problem next
                    next_layer = layer
                    s_next = copy.copy(s_li)
                    for higher_layer in range(layer, -1, -1):
                        next_layer = higher_layer
                        if next_layer == 0:
                            POMDP_key = 'all'
                            break
                        elif next_layer > 0:
                            POMDP_key = self.var_lower_to_var_top(next_layer, s_next['xa'])
                        for s_terminal_next in self.POMDPs[next_layer][POMDP_key].s_terminals:
                            different_states = False
                            for var_name in key_list:
                                if var_name in s_terminal_next.keys():
                                    var_val = s_terminal_next[var_name]
                                    if var_val[0] == '*' or len(var_val) > 1:
                                        continue
                                    if var_val[0] != s_next[var_name]:
                                        different_states = True
                                        break
                            if not different_states:
                                break
                        if different_states:
                            break
                        for var_name, var_val in list(s_next.items()):
                            s_next[var_name] = self.var_lower_to_var_top(layer_lower=next_layer, var_lower=var_val)
                        next_layer -= 1
                    # create belief_enumerated
                    if next_layer == layer:
                        belief_enumerated_next = self.get_terminal_belief_for_every_state_next(
                            next_layer, s_next, self.b0_layers[next_layer],
                            self.POMDPs[next_layer][POMDP_key].x_items_values,
                            self.POMDPs[next_layer][POMDP_key].z_values, x_items_values_li)
                    else:
                        belief_enumerated_next = self.get_terminal_belief_for_every_state_top(
                            next_layer, layer, self.b0_layers[next_layer], s_next,
                            self.POMDPs[next_layer][POMDP_key].x_items_values,
                            self.POMDPs[next_layer][POMDP_key].z_values, b0_li=self.b0_layers[layer],
                            x_items_values_li=x_items_values_li)

                    # read Values from the terminal state POMDP
                    V_best = -1000000
                    sel = self.get_alpha_vectors_selection(s_next, self.POMDPs[next_layer][POMDP_key].xa_values,
                                                           self.POMDPs[next_layer][POMDP_key].x_items_values,
                                                           self.POMDPs[next_layer][POMDP_key].z_values,
                                                           self.POMDPs[next_layer][POMDP_key].alpha_vectors_attrib)
                    for idx, vector in enumerate(self.POMDPs[next_layer][POMDP_key].alpha_vectors[sel]):
                        V_p = np.dot(vector, belief_enumerated_next)
                        if V_p > V_best:
                            V_best = V_p
                    value_table_string += ' {}'.format(V_best + V_penalty)
                else:
                    # STEP 2a: convert s_li to s_top
                    s_top = copy.copy(s_li)
                    top_layer = layer - 1
                    for higher_layer in range(layer - 1, -1, -1):
                        top_layer = higher_layer
                        # set s_top
                        for var_name, var_val in list(s_top.items()):
                            s_top[var_name] = self.var_lower_to_var_top(layer_lower=higher_layer + 1, var_lower=var_val)
                        # get POMDP key
                        if higher_layer == 0:
                            POMDP_key = 'all'
                            break
                        # POMDP_key = self.var_lower_to_var_top(higher_layer, self.s_layers[higher_layer]['xa'])
                        POMDP_key = self.var_lower_to_var_top(higher_layer, s_top['xa'])
                        # if the key does not exist, check if POMDP_key is a terminal state of another POMDP problem
                        if POMDP_key not in self.POMDPs[higher_layer].keys():
                            is_new_key = True
                            for key in self.POMDPs[higher_layer].keys():
                                for s_terminal in self.POMDPs[higher_layer][key].s_terminals:
                                    if POMDP_key == self.var_lower_to_var_top(higher_layer, s_terminal['xa'][0]):
                                        POMDP_key = key
                                        is_new_key = False
                                        break
                                if not is_new_key:
                                    break
                        # check if s_top is not a terminal state in the layer above
                        for s_terminal_top in self.POMDPs[higher_layer][POMDP_key].s_terminals:
                            different_states = False
                            for var_name, var_val in s_terminal_top.items():
                                if var_val[0] == '*' or len(var_val) > 1:
                                    continue
                                if var_val[0] != s_top[var_name]:
                                    different_states = True
                                    break
                            if not different_states:
                                break
                        if different_states:
                            break
                    # STEP 2b: create b_top from s_top
                    belief_top_enumerated = self.get_terminal_belief_for_every_state_top(
                        top_layer, layer, self.b0_layers[top_layer], s_top,
                        self.POMDPs[top_layer][POMDP_key].x_items_values,
                        self.POMDPs[top_layer][POMDP_key].z_values, self.b0_layers[layer], x_items_values_li)
                    # STEP 2c: read Value of top layer with b_top and set as reward
                    V_best = -1000000
                    sel = self.get_alpha_vectors_selection(s_top, self.POMDPs[top_layer][POMDP_key].xa_values,
                                                           self.POMDPs[top_layer][POMDP_key].x_items_values,
                                                           self.POMDPs[top_layer][POMDP_key].z_values,
                                                           self.POMDPs[top_layer][POMDP_key].alpha_vectors_attrib)
                    for idx, vector in enumerate(self.POMDPs[top_layer][POMDP_key].alpha_vectors[sel]):
                        V_p = np.dot(vector, belief_top_enumerated)
                        if V_p > V_best:
                            V_best = V_p
                    value_table_string += ' {}'.format(V_best + V_penalty)
            # change s_li to next state
            keys_all = ['xa'] + list(range(0, self.nr_of_items))
            keys = []
            for key in keys_all:
                if key in s_li.keys():
                    keys.append(key)
            keys.reverse()
            for key in keys:
                if s_li[key] == var_values_li[key][-1]:
                    s_li[key] = var_values_li[key][0]
                else:
                    next_idx = var_values_li[key].index(s_li[key]) + 1
                    s_li[key] = var_values_li[key][next_idx]
                    break

        return value_table_string

    def get_terminal_belief_for_every_state_top(self, top_layer, layer_li, b0_top, s_top, x_items_values_top,
                                                z_values_top, b0_li, x_items_values_li):
        s_top_copy = copy.copy(s_top)
        # remove variables from s_top_copy that are not in z_values_top
        for var_name in s_top.keys():
            if var_name not in z_values_top.keys():
                del s_top_copy[var_name]
        # add variables to s_top that are not in s_li
        for item_idx in range(self.nr_of_items):
            if item_idx in x_items_values_top.keys() and item_idx in z_values_top.keys() and item_idx not in s_top_copy.keys():
                s_top_copy[item_idx] = -1
        b_of_s_top = []
        key_list = list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in s_top_copy.keys():
                b_of_var = []
                for idx, var_val in enumerate(x_items_values_top[var_name]):
                    if s_top_copy[var_name] != -1 and s_top_copy[var_name] != 'not_here' and s_top_copy[
                        var_name] == var_val:
                        b_of_var.append(1.0)
                    elif s_top_copy[var_name] != -1 and s_top_copy[var_name] != 'not_here' and s_top_copy[
                        var_name] != var_val:
                        b_of_var.append(0.0)
                    elif s_top_copy[var_name] == -1 or s_top_copy[var_name] == 'not_here' and (
                            var_val == 'agent' or var_val == 'goal'):
                        b_of_var.append(0.0)
                    elif s_top_copy[var_name] == -1 or s_top_copy[var_name] == 'not_here':
                        if var_val == 'not_here':
                            b_top_not_here = 0
                            for xi_top in self.xa_values_all_layers[top_layer]:
                                if xi_top not in x_items_values_top[var_name]:
                                    b_xi_top = b0_top.get_aggregated_belief(node_nr=xi_top)[var_name]
                                    # WARNING: if below rounding rule is changed, need to change it in
                                    # "write_initial_belief_function_li", "get_belief_for_every_state",
                                    # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                                    if top_layer >= 0:
                                        if b_xi_top > self.b0_threshold:
                                            b_xi_top = 1.0
                                        elif b_xi_top < 1 - self.b0_threshold:
                                            b_xi_top = 0.0
                                    b_top_not_here += b_xi_top
                            b_of_var.append(b_top_not_here)
                        else:
                            b_of_xi = 0.0
                            # xi_values_li_of_xi_top = self.get_subnodes(node=var_val, layer=top_layer)
                            if top_layer + 1 != layer_li:
                                debug = True
                            xi_values_li_of_xi_top = self.get_subnodes_over_multiple_layers(
                                top_node=var_val, top_layer=top_layer, bottom_layer=layer_li)
                            for xi_li in x_items_values_li[var_name]:
                                if xi_li in xi_values_li_of_xi_top:
                                    b_xi_li = b0_li.get_aggregated_belief(node_nr=xi_li)[var_name]
                                    b_of_xi += b_xi_li
                            b_of_xi_top = b0_top.get_aggregated_belief(node_nr=var_val)[var_name] - b_of_xi
                            if top_layer >= 0:
                                if b_of_xi_top > self.b0_threshold:
                                    b_of_xi_top = 1.0
                                elif b_of_xi_top < 1.0 - self.b0_threshold:
                                    b_of_xi_top = 0.0
                            b_of_var.append(b_of_xi_top)
                # normalize row
                row_sum = sum(b_of_var)
                if row_sum > 0:
                    for idx, var_val in enumerate(x_items_values_top[var_name]):
                        b_of_var[idx] /= row_sum
                else:
                    debug = True
                b_of_s_top.append(b_of_var)
        nr_of_item_states = 1
        for var_name in s_top_copy.keys():
            nr_of_item_states *= len(x_items_values_top[var_name])
        belief_top_enumerated = np.zeros(nr_of_item_states)
        var_indices = []
        key_list = list(range(0, self.nr_of_items))
        for var_key in key_list:
            if var_key in s_top_copy.keys():
                var_indices.append(var_key)
        for s_j in range(len(belief_top_enumerated)):
            b = 1.0
            s_j_copy = s_j
            for item_idx in range(len(var_indices)):
                nr_of_xj_values = 1
                for idx in range(len(var_indices) - 1, item_idx, -1):
                    nr_of_xj_values *= len(x_items_values_top[var_indices[idx]])
                xj_top_idx = int(s_j_copy / nr_of_xj_values)
                s_j_copy = s_j_copy % nr_of_xj_values
                b *= b_of_s_top[item_idx][xj_top_idx]
            belief_top_enumerated[s_j] = b
        return belief_top_enumerated

    def get_terminal_belief_for_every_state_next(self, layer, s_li, b0_next, x_items_values_li_next, z_values_li_next,
                                                 x_items_values_li_curr):
        s_li_copy = copy.copy(s_li)
        # remove variables from s_li_copy that are not in z_values_li_next
        for var_name in s_li.keys():
            if var_name not in z_values_li_next.keys():
                del s_li_copy[var_name]
        b_of_s_next = []
        key_list = list(range(0, self.nr_of_items))
        for var_name in key_list:
            if var_name in s_li_copy.keys():
                b_of_var = []
                for idx, var_val in enumerate(x_items_values_li_next[var_name]):
                    if s_li_copy[var_name] != 'not_here' and s_li_copy[var_name] == var_val:
                        b_of_var.append(1.0)
                    elif s_li_copy[var_name] != 'not_here' and var_val != 'not_here' and s_li_copy[var_name] != var_val:
                        b_of_var.append(0.0)
                    elif s_li_copy[var_name] != 'not_here' and var_val == 'not_here':
                        if s_li_copy[var_name] not in x_items_values_li_next[var_name]:
                            b_of_var.append(1.0)
                        else:
                            b_of_var.append(0.0)
                    elif s_li_copy[var_name] == 'not_here':
                        if var_val != 'not_here' and (
                                var_val in x_items_values_li_curr[var_name] or var_val == 'agent' or var_val == 'goal'):
                            b_of_var.append(0.0)
                        elif var_val != 'not_here':
                            b_next = b0_next.get_aggregated_belief(node_nr=var_val)[var_name]
                            # WARNING: if below if condition is changed, need to change it in
                            # "write_initial_belief_function_li", "get_belief_for_every_state",
                            # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                            if layer >= 0:
                                if b_next > self.b0_threshold:
                                    b_next = 1.0
                                elif b_next < 1 - self.b0_threshold:
                                    b_next = 0.0
                            b_of_var.append(b_next)
                        else:
                            b_of_xj = 0
                            for xa_lj in self.xa_values_all_layers[layer]:
                                if xa_lj not in x_items_values_li_curr[var_name] and xa_lj not in \
                                        x_items_values_li_next[var_name]:
                                    b_next = b0_next.get_aggregated_belief(node_nr=xa_lj)[var_name]
                                    # WARNING: if below if condition is changed, need to change it in
                                    # "write_initial_belief_function_li", "get_belief_for_every_state",
                                    # "get_belief_for_every_state_next", "get_belief_for_every_state_top",
                                    if layer >= 0:
                                        if b_next > self.b0_threshold:
                                            b_next = 1.0
                                        elif b_next < 1 - self.b0_threshold:
                                            b_next = 0.0
                                    b_of_xj += b_next
                            b_of_var.append(b_of_xj)
                # normalize row
                row_sum = sum(b_of_var)
                if row_sum == 0:
                    debug = True
                else:
                    for idx, var_val in enumerate(x_items_values_li_next[var_name]):
                        b_of_var[idx] /= row_sum
                b_of_s_next.append(b_of_var)

        nr_of_item_states = 1
        for var_name in s_li_copy.keys():
            nr_of_item_states *= len(x_items_values_li_next[var_name])
        belief_next_enumerated = np.zeros(nr_of_item_states)
        var_indices = []
        key_list = list(range(0, self.nr_of_items))
        for var_key in key_list:
            if var_key in s_li_copy.keys():
                var_indices.append(var_key)
        for s_j in range(len(belief_next_enumerated)):
            b = 1.0
            s_j_copy = s_j
            for item_idx in range(len(var_indices)):
                nr_or_xj_values = 1
                for idx in range(len(var_indices) - 1, item_idx, -1):
                    nr_or_xj_values *= len(x_items_values_li_next[var_indices[idx]])
                xj_next_idx = int(s_j_copy / nr_or_xj_values)
                s_j_copy = s_j_copy % nr_or_xj_values
                b *= b_of_s_next[item_idx][xj_next_idx]
            belief_next_enumerated[s_j] = b
        return belief_next_enumerated
