# from scripts import run_experiment

import math
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm

from src.auxiliary_files.grid import Grid
from src.auxiliary_files.item import Item
from src.multi_scale_search.agents import AgentMultiScaleM1
from src.multi_scale_search.agents import AgentMultiScaleM2
# from agent_MultiScale_M3 import AgentMultiScaleM3
from src.auxiliary_files.rectangle import Rectangle
from src.low_fidelity_simulation.worldmodel import WorldModel
from src.multi_scale_search import auxiliary_functions
import config
# from agent_b1 import AgentB1
from src.multi_scale_search.agents import AgentFLAT
from src.multi_scale_search.belief import Belief
from src.multi_scale_search.core import Agent
from src.low_fidelity_simulation.low_level_controller import ControllerLFS


class LowFidelitySimulation:

    def __init__(self):
        self.agent_type = 'FLAT'
        self.environment = 'small'
        self.world_grid, self.agent_grid = Grid(), Grid()
        self.rec_env = []
        self.lines_env = []
        self.config0 = [-1, -1, 0]
        self.agent = Agent()
        self.world = WorldModel()
        self.items = []
        self.task = {}
        self.history = {}
        self.history_x_values, self.history_y_values, self.history_theta_values, self.history_actions = {}, {}, {}, {}
        self.history_carried = {}
        self.history_agent_cells = {}
        self.history_belief = {}
        self.k = 0  # simulation step
        self.max_it = 4000
        self.start_time = -1
        self.initialized_grids = False
        self.initialized = False
        self.initial_belief = np.zeros((1, 4))
        self.belief_spot_width = 0.1
        self.visualisation_ratio = 1
        self.b0_threshold = 0.999
        self.timeout_time = 5.0
        self.print_every_timestep = True
        self.controller = ControllerLFS(self.agent_grid)

    def reinitialize(self):
        # clean histories
        self.k = 0
        self.history, self.history_x_values, self.history_y_values, self.history_theta_values,\
        self.history_actions = {}, {}, {}, {}, {}
        # initialize agent and world grid
        self.initialize_grids()
        # initialize agent
        node_mapping = []
        nr_of_layers = 1
        if self.environment == 'small':
            if self.agent_type in config.multiscale_agent_types:
                nr_of_layers = 2
                node_mapping = [{0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]}]
        elif self.environment == 'big':
            if self.agent_type in config.multiscale_agent_types:
                nr_of_layers = 3
                node_mapping = [{0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15]},
                                {0: [24, 25, 26, 27], 1: [0, 1, 2], 2: [3, 4, 5], 3: [28, 29, 30, 31],
                                 4: [32, 33, 34, 35], 5: [6, 7, 8], 6: [9, 10, 11], 7: [36, 37, 38, 39],
                                 8: [40, 41, 42, 43], 9: [12, 13, 14], 10: [15, 16, 17], 11: [44, 45, 46, 47],
                                 12: [48, 49, 50, 51], 13: [18, 19, 20], 14: [21, 22, 23], 15: [52, 53, 54, 55]}]

        item_names = []
        for item in self.items:
            item_names += [item.item_type]
        if self.agent_type == 'FLAT':
            self.agent = AgentFLAT(self.agent_grid, self.config0, rec_env=self.rec_env, item_names=item_names,
                                   environment=self.environment)
        elif self.agent_type == 'MultiScaleAP':
            self.agent = AgentMultiScaleM1(self.agent_grid, self.config0, nr_of_layers, node_mapping=node_mapping,
                                           rec_env=self.rec_env, item_names=item_names, environment=self.environment)
        elif self.agent_type == 'MultiScaleAVP':
            self.agent = AgentMultiScaleM2(self.agent_grid, self.config0, nr_of_layers, node_mapping=node_mapping,
                                           rec_env=self.rec_env, item_names=item_names, environment=self.environment)

        # if self.agent_type == 'b1':
        #     self.agent = AgentB1(self.agent_grid, self.config0)

        #     self.agent = AgentFLAT(self.agent_grid, self.config0, self.rec_env, item_names)
        # elif self.agent_type == 'DualScaleM1':
        #     item_names = []
        #     for item in self.items:
        #         item_names += [item.item_type]
        #     self.agent = AgentMultiScaleM1(self.agent_grid, self.config0, nr_of_layers=2, node_mapping=node_mapping,
        #                                    rec_env=self.rec_env, item_names=item_names)
        # elif self.agent_type == 'DualScaleM2':
        #     item_names = []
        #     for item in self.items:
        #         item_names += [item.item_type]
        #     self.agent = AgentMultiScaleM2(self.agent_grid, self.config0, nr_of_layers=2, node_mapping=node_mapping,
        #                                    rec_env=self.rec_env, item_names=item_names)
        # elif self.agent_type == 'DualScaleM3':
        #     item_names = []
        #     for item in self.items:
        #         item_names += [item.item_type]
        #     self.agent = AgentMultiScaleM3(self.agent_grid, self.config0, nr_of_layers=2, node_mapping=node_mapping,
        #                                    rec_env=self.rec_env, item_names=item_names)
        # elif self.agent_type == 'MultiScaleM1':
        #     item_names = []
        #     for item in self.items:
        #         item_names += [item.item_type]
        #     self.agent = AgentMultiScaleM1(self.agent_grid, self.config0, nr_of_layers=3, node_mapping=node_mapping,
        #                                    rec_env=self.rec_env, item_names=item_names)
        # elif self.agent_type == 'MultiScaleM2':
        #     item_names = []
        #     for item in self.items:
        #         item_names += [item.item_type]
        #     self.agent = AgentMultiScaleM2(self.agent_grid, self.config0, nr_of_layers=3, node_mapping=node_mapping,
        #                                    rec_env=self.rec_env, item_names=item_names)
        # elif self.agent_type == 'MultiScaleM3':
        #     item_names = []
        #     for item in self.items:
        #         item_names += [item.item_type]
        #     self.agent = AgentMultiScaleM3(self.agent_grid, self.config0, nr_of_layers=3, node_mapping=node_mapping,
        #                                    rec_env=self.rec_env, item_names=item_names)
        # initialize the world
        self.world = WorldModel(self.world_grid, self.config0)
        # add items to the world_grid
        for item in self.items:
            self.world.add_item_to_grid(item.item_type, item.x, item.y)
        if self.agent_type != 'b1':
            # set belief
            self.agent.set_belief(x_list=self.initial_belief[:, 0], y_list=self.initial_belief[:, 1],
                                  b_list=self.initial_belief[:, 2], b_sigma=self.belief_spot_width,
                                  item_nr_list=self.initial_belief[:, 3])
        # set timeout time
        self.agent.set_timeout_time(self.timeout_time)
        if self.agent_type in config.POMDP_agent_types:
            self.agent.set_b0_threshold(self.b0_threshold)
        # set task
        self.agent.set_task(self.task.copy())
        self.world.set_task(self.task.copy())
        # perform very first observation (no action)
        self.agent.start_solving()
        o_agent_conf, o_cells, o_carry, o_fin_tasks, world_time = self.world.evaluate_observation_model()
        self.agent.update_carry(o_carry=o_carry)
        self.agent.update_pose(o_x=o_agent_conf[0], o_y=o_agent_conf[1], o_theta=o_agent_conf[2])
        self.agent.interpret_observations(o_cells)  # world_time
        self.initialized = True

    def run_sim(self):
        if not self.initialized:
            return 'E:init'
        # START SIMULATION, ITERATION = 0
        while 0 < len(self.world.open_tasks) and self.k < self.max_it:
            # get current state for history
            state = (self.world.agent_x, self.world.agent_y, math.degrees(self.world.agent_theta),
                     self.world.agent_carries, self.world.finished_tasks[:])
            # agent chooses goal_reference
            goal_ref = self.agent.choose_action()
            # let controller steer the robot
            robot_action = self.controller.get_control_input(self.world.agent_x, self.world.agent_y,
                                                             self.world.agent_theta, goal_ref)
            # apply action and get observation
            o_agent_conf, o_cells, o_carry, o_fin_tasks, world_time = self.world.evaluate_action_get_observation(robot_action)
            # agent interprets observation
            self.agent.update_carry(o_carry)
            self.agent.update_pose(o_x=o_agent_conf[0], o_y=o_agent_conf[1], o_theta=o_agent_conf[2])
            self.agent.interpret_observations(o_cells)

            # save relevant information of this timestep for the documentation
            self.history[self.k] = (self.world.time, state, robot_action)
            self.history_x_values[self.k], self.history_y_values[self.k], self.history_theta_values[self.k] = \
                state[0], state[1], state[2]
            if self.agent_type == 'b1' and self.visualisation_ratio != 'none':
                self.history_agent_cells[self.k] = self.agent.grid.get_seen_as_2Dlist()
            elif self.agent_type != 'b1' and self.visualisation_ratio != 'none':
                self.history_belief[self.k] = self.agent.belief.create_grid_data(
                    nr_of_cells_x=int(self.world.grid.total_width * self.visualisation_ratio),
                    nr_of_cells_y=int(self.world.grid.total_height * self.visualisation_ratio),
                    world_width=self.world.grid.total_width,
                    world_height=self.world.grid.total_height)
            if self.agent.replanning:
                self.history_actions[self.k] = 'replanning'
                self.agent.set_replanning(False)
            else:
                self.history_actions[self.k] = robot_action
            self.history_carried = state[3]
            # print information of current timestep
            if self.print_every_timestep:
                print(
                    'k = {},  time={},  ag_conf_old={},   ag_carry={},  deliv. items={},  action_value={}'.format(
                        self.k, self.history[self.k][0], (state[0], state[1], state[2]), state[3], state[4],
                        self.history[self.k][2]))
            self.k += 1

        sol_quality = self.world.time
        comp_time = self.agent.computation_time
        comp_time_avr, comp_time_layers, comp_time_layers_avr = 0, [], []
        print('Solution Quality (simulation time) = {}'.format(sol_quality))
        print('total computation time = {}'.format(comp_time))
        if self.agent_type == 'FLAT':
            comp_time_avr = self.agent.computation_time / len(self.agent.subroutine_actions_history.keys())
            print('average total comp. time = {}'.format(comp_time_avr))
        elif self.agent_type in config.multiscale_agent_types:
            comp_time_avr = comp_time / len(self.agent.subroutine_actions_history[-1].keys())
            comp_time_layers = self.agent.computation_time_layers
            for layer, comp_time_layer in enumerate(comp_time_layers):
                print('computation time layer {} = {}\n'.format(layer + 1, comp_time_layer))
            for layer, comp_time_layer in enumerate(comp_time_layers):
                print('averaeg computation time layer {} = {}\n'.format(
                    layer + 1, comp_time_layer / len(self.agent.subroutine_actions_history[-1].keys())))
                comp_time_layers_avr.append(comp_time_layer / len(self.agent.subroutine_actions_history[-1].keys()))

        return self.k, sol_quality, comp_time, comp_time_avr, comp_time_layers, comp_time_layers_avr, \
               self.agent.computation_time_for_each_action

    def draw_documentation(self):
        plt.close('all')
        self.draw_animation()
        # draw the end picture separately
        self.draw_entire_game()
        plt.show()

    # Dear reader, I hope you never have to fix/change something within this function
    def draw_animation(self):
        fig1 = plt.figure()
        fig1.suptitle('task: ' + self.convert_task_to_string(self.task))
        axes = []
        nr_of_subplots = 1
        if self.agent_type != 'b1':
            nr_of_subplots = len(self.items)
        for idx in range(nr_of_subplots):
            axes.append(plt.subplot(1, nr_of_subplots, 1 + idx))
            axes[-1].set_xlim(-0.5, self.world.grid.total_width + 0.5)
            axes[-1].set_ylim(-0.5, self.world.grid.total_width + 0.5)
            axes[-1].invert_yaxis()
            axes[-1].set_aspect('equal')
            if self.agent_type != 'b1':
                axes[-1].set_title('b(x{})'.format(idx))
        # draw environment
        artists_env = []
        for ax in axes:
            artist_row = []
            for line_idx in range(0, len(self.lines_env)):
                artist_row.append(ax.plot([], [], color='k', linewidth=2, zorder=5)[0])
            artists_env.append(artist_row)
        artists_env = np.array(artists_env)
        # draw items
        artists_items = []
        artists_goals = []
        for ax in axes:
            artists_items_row = []
            artists_goals_row = []
            for item in self.items:
                c = 'black'
                if item.item_type == 'mug':
                    c = 'magenta'
                elif item.item_type == 'plate':
                    c = 'green'
                elif item.item_type == 'milk':
                    c = 'blue'
                markersize = max(2, 5 - len(self.items))
                artists_items_row.append(ax.plot([], [], marker='s', markersize=markersize, color=c, zorder=4)[0])
                artists_goals_row.append(ax.plot([], [], marker='h', markersize=markersize, color=c, zorder=4)[0])
            artists_items.append(artists_items_row)
            artists_goals.append(artists_goals_row)
        artists_items = np.array(artists_items)
        artists_goals = np.array(artists_goals)

        line_ag, line_vc, line_look_around, line_pickup, line_release, line_replan = [], [], [], [], [], []
        for ax in axes:
            line_ag.append(ax.plot([], [], lw=1, color='k', zorder=2)[0])
            line_vc.append(ax.plot([], [], lw=1, color='r', zorder=1)[0])  # lines of the viewing cone
            line_look_around.append(ax.plot([], [], color='r', marker='*', linestyle='None', zorder=2)[0])
            line_pickup.append(ax.plot([], [], color='r', marker='o', linestyle='None', zorder=2)[0])
            line_release.append(ax.plot([], [], color='r', marker='^', linestyle='None', zorder=2)[0])
            line_replan.append(ax.plot([], [], color='brown', marker='v', linestyle='None', zorder=2)[0])

        if self.agent_type == 'b1':
            colors = ['white', 'grey']
            cm = LinearSegmentedColormap.from_list('name1', colors, N=2)
            colormap_grid = ax.imshow([[]], interpolation='none', cmap=cm,
                                      extent=[0, self.agent.grid.total_width, 0, self.agent.grid.total_height],
                                      zorder=0)
        else:
            colormap_belief = []
            for ax in axes:
                colormap_belief.append(
                    ax.imshow([[]], interpolation='none', norm=LogNorm(vmin=10 ** (-10), vmax=1.0),
                              extent=[0, self.world_grid.total_width, 0, self.world_grid.total_height], zorder=0))
        if self.agent_type == 'b1':
            pickup_actions = ['pickup_mug', 'pickup_plate', 'pickup_milk']
            pickup_keys = [key for key, value in self.history_actions.items() if value in pickup_actions]
            release_keys = [key for key, value in self.history_actions.items() if value == 'release_item']
            look_around_keys = []
            replanning_keys = [key for key, value in self.history_actions.items() if value == 'replanning']
        else:
            if self.agent_type == 'FLAT':
                action_history = self.agent.subroutine_actions_history
            elif self.agent_type in config.multiscale_agent_types:
                action_history = self.agent.subroutine_actions_history[-1]
            pickup_keys = [key for key, value in action_history.items() if value[0:6] == 'pickup']
            release_keys = [key for key, value in action_history.items() if value == 'release']
            look_around_keys = [key for key, value in action_history.items() if value == 'look_around']
            replanning_keys = []

        def init():
            # environment and items
            for row in artists_env:
                for idx, artist in enumerate(row):
                    artist.set_data(self.lines_env[idx][0], self.lines_env[idx][1])
            for row in artists_items:
                for idx, artist in enumerate(row):
                    artist.set_data(self.items[idx].x, self.items[idx].y)
            for row in artists_goals:
                for idx, artist in enumerate(row):
                    artist.set_data(self.task[self.items[idx].item_type][0], self.task[self.items[idx].item_type][1])
            # agent and actions
            for idx in range(len(axes)):
                line_ag[idx].set_data([self.history_x_values[0]], [self.history_y_values[0]])
                line_pickup[idx].set_data([], [])
                line_release[idx].set_data([], [])
                line_replan[idx].set_data([], [])
                line_vc[idx].set_data([], [])
            if self.agent_type == 'b1':
                colormap_grid.set_data([[]])
                artist_list = list(artists_env.flatten()) + list(artists_items.flatten()) + list(
                    artists_goals.flatten()) + line_ag + line_pickup + line_release + \
                              line_replan + line_vc + [colormap_grid]
            else:
                for idx in range(len(axes)):
                    colormap_belief[idx].set_data([[]])
                artist_list = list(artists_env.flatten()) + list(artists_items.flatten()) + list(
                    artists_goals.flatten()) + line_ag + line_look_around + line_pickup + line_release + \
                              line_vc + colormap_belief

            return artist_list

        def animate(i):
            # colormap
            if self.agent_type == 'b1':
                N_x, N_y = self.agent.grid.nr_of_cells_x, self.agent.grid.nr_of_cells_y
                data = np.ones((N_y, N_x))
                for v in range(0, N_y):
                    for u in range(0, N_x):
                        data[N_y - 1 - v, u] = self.history_agent_cells[i][v][u]
                colormap_grid.set_data(data)
            else:
                for idx, cm_b in enumerate(colormap_belief):
                    cm_b.set_data(self.history_belief[i][idx])
            # environment and items
            for row in artists_env:
                for idx, artist in enumerate(row):
                    artist.set_data(self.lines_env[idx][0], self.lines_env[idx][1])
            for row in artists_items:
                for idx, artist in enumerate(row):
                    artist.set_data(self.items[idx].x, self.items[idx].y)
            for row in artists_goals:
                for idx, artist in enumerate(row):
                    artist.set_data(self.task[self.items[idx].item_type][0],
                                    self.task[self.items[idx].item_type][1])

            # agent and actions
            pickup_x_values, pickup_y_values = [], []
            release_x_values, release_y_values = [], []
            look_around_x_values, look_around_y_values = [], []
            replanning_x_values, replanning_y_values = [], []
            for k in range(i + 1):
                if k in pickup_keys:
                    pickup_x_values.append(self.history_x_values[k])
                    pickup_y_values.append(self.history_y_values[k])
                elif k in release_keys:
                    release_x_values.append(self.history_x_values[k])
                    release_y_values.append(self.history_y_values[k])
                elif k in look_around_keys:
                    look_around_x_values.append(self.history_x_values[k])
                    look_around_y_values.append(self.history_y_values[k])
                elif k in replanning_keys:
                    replanning_x_values.append(self.history_x_values[k])
                    replanning_y_values.append(self.history_y_values[k])
            for idx in range(len(axes)):
                line_ag[idx].set_data(list(self.history_x_values.values())[0:i],
                                      list(self.history_y_values.values())[0:i])
                line_pickup[idx].set_data(pickup_x_values, pickup_y_values)
                line_release[idx].set_data(release_x_values, release_y_values)
                line_look_around[idx].set_data(look_around_x_values, look_around_y_values)
                line_replan[idx].set_data(replanning_x_values, replanning_y_values)

            # draw viewing cone
            # get transformation function of robot:
            theta_i = self.history_theta_values[i] * math.pi / 180
            trans_matrix = np.array([[math.cos(theta_i), math.sin(theta_i), self.history_x_values[i]],
                                     [-math.sin(theta_i), math.cos(theta_i), self.history_y_values[i]],
                                     [0, 0, 1]])
            x_values = [config.robot_viewing_cone[0][0][0]]
            y_values = [config.robot_viewing_cone[0][0][1]]
            for rays in config.robot_viewing_cone:
                x_values.append(rays[1][0])
                y_values.append(rays[1][1])
            x_values.append(config.robot_viewing_cone[-1][0][0])
            y_values.append(config.robot_viewing_cone[-1][0][1])
            x_values.append(config.robot_viewing_cone[0][0][0])
            y_values.append(config.robot_viewing_cone[0][0][1])
            # transform all points
            x_values_trans, y_values_trans = [], []
            for i in range(len(x_values)):
                p_h = np.array([x_values[i], y_values[i], 1])
                p_trans = np.dot(trans_matrix, p_h)
                x_values_trans.append(p_trans[0])
                y_values_trans.append(p_trans[1])
            # set data
            for idx in range(len(axes)):
                line_vc[idx].set_data(x_values_trans, y_values_trans)
            # return everything as one list
            if self.agent_type == 'b1':
                artist_list = list(artists_env.flatten()) + list(artists_items.flatten()) + list(
                    artists_goals.flatten()) + line_ag + line_pickup + line_release + \
                              line_replan + line_vc + [colormap_grid]
            else:
                artist_list = list(artists_env.flatten()) + list(artists_items.flatten()) + list(
                    artists_goals.flatten()) + line_ag + line_look_around + line_pickup + line_release + line_vc + \
                              colormap_belief
            return artist_list

        anim = animation.FuncAnimation(fig1, animate, init_func=init, frames=len(self.history_x_values.keys()),
                                       interval=100, repeat_delay=500, blit=True)
        anim.save('animation.mp4')

    def draw_entire_game(self):
        fig = plt.figure()
        ax = plt.axes(xlim=(-0.5, self.world.grid.total_width + 0.5), ylim=(-0.5, self.world.grid.total_width + 0.5))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.invert_yaxis()
        # draw environment
        for line in self.lines_env:
            ax.plot(line[0], line[1], color='black', linewidth=2)
        # draw items
        for item in self.items:
            c = 'black'
            if item.item_type == 'mug':
                c = 'magenta'
            elif item.item_type == 'plate':
                c = 'green'
            elif item.item_type == 'milk':
                c = 'blue'
            markersize = max(2, 5 - len(self.items))
            ax.plot(item.x, item.y, marker='s', markersize=markersize, color=c, label=item.item_type)
            ax.plot(self.task[item.item_type][0], self.task[item.item_type][1], marker='h', markersize=markersize,
                    color=c, label=item.item_type + '-goal')
        ax.set_title('task: ' + self.convert_task_to_string(self.task))

        ax.plot(list(self.history_x_values.values()), list(self.history_y_values.values()), lw=1, color='k')

        if self.agent_type == 'b1':
            pickup_actions = ['pickup_mug', 'pickup_plate', 'pickup_milk']
            pickup_keys = [key for key, value in self.history_actions.items() if value in pickup_actions]
            release_keys = [key for key, value in self.history_actions.items() if value == 'release_item']
            look_around_keys = []
            replanning_keys = [key for key, value in self.history_actions.items() if value == 'replanning']
        else:
            if self.agent_type == 'FLAT':
                action_history = self.agent.subroutine_actions_history
            elif self.agent_type in config.multiscale_agent_types:
                action_history = self.agent.subroutine_actions_history[-1]
            pickup_keys = [key for key, value in action_history.items() if value[0:6] == 'pickup']
            release_keys = [key for key, value in action_history.items() if value == 'release']
            look_around_keys = [key for key, value in action_history.items() if value == 'look_around']
            replanning_keys = []

        pickup_x_values, pickup_y_values = [], []
        release_x_values, release_y_values = [], []
        look_around_x_values, look_around_y_values = [], []
        replanning_x_values, replanning_y_values = [], []
        for k in range(len(self.history_x_values)):
            if k in pickup_keys:
                pickup_x_values.append(self.history_x_values[k])
                pickup_y_values.append(self.history_y_values[k])
            elif k in release_keys:
                release_x_values.append(self.history_x_values[k])
                release_y_values.append(self.history_y_values[k])
            elif k in look_around_keys:
                look_around_x_values.append(self.history_x_values[k])
                look_around_y_values.append(self.history_y_values[k])
            elif k in replanning_keys:
                replanning_x_values.append(self.history_x_values[k])
                replanning_y_values.append(self.history_y_values[k])

        ax.plot(pickup_x_values, pickup_y_values, 'ro', linestyle='None', label='pickup action')
        ax.plot(release_x_values, release_y_values, 'r^', linestyle='None', label='release action')
        if self.agent_type in config.POMDP_agent_types:
            ax.plot(look_around_x_values, look_around_y_values, 'r*', linestyle='None', label='look_around action')
        elif self.agent_type == 'b1':
            ax.plot(replanning_x_values, replanning_y_values, color='brown', marker='v', linestyle='None',
                    label='replanning')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 6})

    def set_b0_threshold(self, b0_threshold):
        self.b0_threshold = b0_threshold

    def set_print_every_timestep(self, print_every_timestep):
        self.print_every_timestep = print_every_timestep

    def set_timeout_time(self, timeout_time):
        self.timeout_time = timeout_time

    def set_agent_type(self, agent_type):
        self.agent_type = agent_type

    def set_agent_conf0(self, x, y, theta):
        self.config0 = (x, y, theta)

    def set_visualisation_ratio(self, visualisation_ratio):
        self.visualisation_ratio = visualisation_ratio

    def set_env_type(self, environment_type):
        self.environment = environment_type
        self.initialize_grids()

    def set_max_it(self, max_it):
        self.max_it = max_it

    def set_item(self, item_type, in_x, in_y, g_x, g_y):
        if not self.initialized_grids:
            self.initialize_grids()
        # check if in_x, in_y is in free space of room
        in_u, in_v = self.agent_grid.get_cell_indices_by_position(in_x, in_y)
        # check if distance to next item is big enough
        for item in self.items:
            dist = (in_x - item.x) ** 2 + (in_y - item.y) ** 2
            if dist < 1.2 * ((3 * self.world_grid.cell_width) ** 2 + (3 * self.world_grid.cell_height) ** 2):
                return 'E:dist'
        if self.agent_grid.cells[in_v][in_u].value == 'free':
            self.items.append(Item(item_type=item_type, x=in_x, y=in_y))
        else:
            return 'E:item'
        # check if g_x, g_y in free space of room
        g_u, g_v = self.agent_grid.get_cell_indices_by_position(g_x, g_y)
        if self.agent_grid.cells[g_v][g_u].value == 'free':
            self.task[item_type] = (g_x, g_y)
        else:
            return 'E:goal'
        return 'success'

    # belief_spots is a 2D-numpy array with 4 columns, 0=x, 1=y, 2=prob, 3=item_type
    def set_belief(self, belief_spots, belief_spot_width):
        self.initial_belief = belief_spots
        self.belief_spot_width = belief_spot_width

    def delete_items(self):
        self.items = []
        self.task = {}

    def initialize_grids(self):
        if self.environment == 'small':
            # room layout:
            rec_office1 = Rectangle(x0=0, y0=0, width=6, height=4)
            rec_hallway1 = Rectangle(x0=2, y0=4, width=2, height=1.5)
            rec_office2 = Rectangle(x0=0, y0=5.5, width=6, height=6)
            rec_hallway2 = Rectangle(x0=6, y0=7.5, width=1.5, height=2)
            rec_kitchen = Rectangle(x0=7.5, y0=3.5, width=6.5, height=10)
            self.rec_env = [rec_office1, rec_hallway1, rec_office2, rec_hallway2, rec_kitchen]
            # create lines of environment
            self.lines_env = []
            for rec in self.rec_env:
                auxiliary_functions.get_lines(rec, self.lines_env)
            # items
            # create grid for world
            self.world_grid = Grid(nr_of_cells_x=int(140), nr_of_cells_y=int(135), world_width=14, world_height=13.5,
                                   default_value='occupied', grid_name='small')
            # create grid for agent
            default_seen = 1
            self.agent_grid = Grid(nr_of_cells_x=70, nr_of_cells_y=67, world_width=14, world_height=13.4,
                                   default_value='occupied', grid_name='small', default_seen=default_seen)
            # set room layout to world-grid
            self.world_grid.set_region(rec_office1, 'free')
            self.world_grid.set_region(rec_hallway1, 'free')
            self.world_grid.set_region(rec_office2, 'free')
            self.world_grid.set_region(rec_hallway2, 'free')
            self.world_grid.set_region(rec_kitchen, 'free')
            # set room layout to agent-grid
            self.agent_grid.set_region(rec_office1, 'free')
            self.agent_grid.set_region(rec_hallway1, 'free', y1_beh='bigger')
            self.agent_grid.set_region(rec_office2, 'free')
            self.agent_grid.set_region(rec_hallway2, 'free', x1_beh='bigger')
            self.agent_grid.set_region(rec_kitchen, 'free')
            self.initialized_grids = True

        elif self.environment == 'big':
            rec_env = []
            # top 4 small rooms:
            for i in range(4):
                rec_env.append(Rectangle(x0=0.6 + i * (6 + 1), y0=0.6, width=6, height=8))
            # top hallways:
            for i in range(4):
                rec_env.append(Rectangle(x0=0.6 + 2 + i * (1 + 6), y0=0.6 + 8, width=2, height=0.6))
            # large room in the middle
            rec_env.append(Rectangle(x0=0.6, y0=0.6 + 8 + 0.6, width=4 * 6 + 3 * 1, height=8))
            # bottom hallways:
            for i in range(4):
                rec_env.append(Rectangle(x0=0.6 + 2 + i * (1 + 6), y0=0.6 + 8 + 0.6 + 8, width=2, height=0.6))
            # bottom 4 small rooms:
            for i in range(4):
                rec_env.append(Rectangle(x0=0.6 + i * (6 + 1), y0=0.6 + 8 + 0.6 + 8 + 0.6, width=6, height=8))
            self.rec_env = rec_env
            # create lines of environment
            self.lines_env = []
            for rec in self.rec_env:
                auxiliary_functions.get_lines(rec, self.lines_env)
            # create grid for world
            self.world_grid = Grid(nr_of_cells_x=280, nr_of_cells_y=260, world_width=28, world_height=26,
                                   default_value='occupied', grid_name='big')
            # create grid for agent
            default_seen = 1
            self.agent_grid = Grid(nr_of_cells_x=140, nr_of_cells_y=130, world_width=28, world_height=26,
                                   default_value='occupied', grid_name='big', default_seen=default_seen)
            # set room layout to world-grid and agent-grid
            for idx, rec in enumerate(rec_env):
                # print('rec_nr={}, x0={}, y0={}, width={}, height={}\n'.format(idx, rec.x0, rec.y0, rec.width, rec.height))
                self.world_grid.set_region(rec, 'free', y1_beh='bigger')
                self.agent_grid.set_region(rec, 'free', y1_beh='bigger')
            rec_occupied = Rectangle(x0=0.6 + 9, y0=0.6 + 10.6, width=9, height=4)
            self.world_grid.set_region(rec_occupied, 'occupied')
            self.agent_grid.set_region(rec_occupied, 'occupied')
            self.initialized_grids = True
            # config.draw_rectangles(self.rec_env)

        else:
            print('environment type is not valid: {}'.format(self.environment))

        self.controller = ControllerLFS(grid=self.agent_grid)

    def convert_task_to_string(self, task):
        text = ''
        for key in task:
            if text != '':
                text += ', '
            for idx, item in enumerate(self.items):
                if item.item_type == key:
                    text += 'x{}'.format(idx)
            text += '->'
            text += str(task[key])
        return text

    def draw_world(self, items_conf=-1):
        plt.close('all')
        # initialize grids
        if not self.initialized_grids:
            self.initialize_grids()
        # initialize drawing
        fig = plt.figure()
        ax = plt.axes()

        ax.set_xlim(-0.5, self.agent_grid.total_width + 0.5)
        ax.set_ylim(-0.5, self.agent_grid.total_height + 0.5)
        ax.invert_yaxis()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # draw environment
        for line in self.lines_env:
            ax.plot(line[0], line[1], color='black', linewidth=2)
        # draw items
        if items_conf == -1:
            for item in self.items:
                c = 'black'
                if item.item_type == 'mug':
                    c = 'magenta'
                elif item.item_type == 'plate':
                    c = 'green'
                elif item.item_type == 'milk':
                    c = 'blue'
                elif item.item_type == 'donut':
                    c = 'brown'
                markersize = max(2, 5 - len(self.items))
                ax.plot(item.x, item.y, marker='s', color=c, label=item.item_type, markersize=markersize)
                ax.plot(self.task[item.item_type][0], self.task[item.item_type][1], marker='h', color=c,
                        markersize=markersize, label=item.item_type + '-goal')
            ax.set_title('task: ' + self.convert_task_to_string(self.task))
        else:
            task_string = 'task: '
            for item_line in items_conf:
                c = 'black'
                item_type = item_line.cb_type.get()
                x, y = item_line.tb_x.get(), item_line.tb_y.get()
                g_x, g_y = item_line.tb_gx.get(), item_line.tb_gy.get()
                try:
                    x, y = float(x), float(y)
                except:
                    x, y = 0, 0
                try:
                    g_x, g_y = float(g_x), float(g_y)
                except:
                    g_x, g_y = 0, 0
                if item_type == 'mug':
                    c = 'magenta'
                elif item_type == 'plate':
                    c = 'green'
                elif item_type == 'milk':
                    c = 'blue'
                elif item_type == 'donut':
                    c = 'brown'
                markersize = max(2, 5 - len(items_conf))
                ax.plot(x, y, marker='s', color=c, label=item_type, markersize=markersize)
                ax.plot(g_x, g_y, marker='h', color=c, label=item_type + '-goal', markersize=markersize)
                task_string += '{}->({}, {});'.format(item_type, g_x, g_y)
            ax.set_title(task_string)
        # draw agent
        if not self.initialized:
            ax.plot(self.config0[0], self.config0[1], marker='o', color='black', label='start pos.')
        else:
            ax.plot(self.agent.x, self.agent.y, marker='o', color='black')
            # legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), prop={'size': 6})
        plt.show()

    def draw_b0(self, beliefspots_conf, beliefspot_width, items_conf):
        plt.close('all')
        # create rectangles for nodes
        self.initialize_grids()
        # initialize belief
        # map item names to item_nr
        item_map = {}
        for idx, item_line in enumerate(items_conf):
            item_map[item_line.cb_type.get()] = idx
        belief = Belief(total_nr_of_cells=10000, recs_beliefgrids=self.rec_env, nr_of_items=len(items_conf))
        # set belief spots
        fig = plt.figure()
        for key in item_map:
            for b_line in beliefspots_conf:
                if b_line.cb_type.get() == key:
                    x, y, w = float(b_line.tb_x.get()), float(b_line.tb_y.get()), float(b_line.tb_prob.get())
                    belief.add_belief_spot(mu_x=x, mu_y=y, prob=w, sigma=beliefspot_width, item_nr=item_map[key])
            # initialize drawing
            ax = plt.subplot(1, len(items_conf), 1 + item_map[key])
            ax.set_xlim(-0.5, self.agent_grid.total_width + 0.5)
            ax.set_ylim(-0.5, self.agent_grid.total_height + 0.5)
            ax.invert_yaxis()
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.set_title('b(x{})'.format(item_map[key]))
            artist_b0 = ax.imshow([[]], interpolation='none', norm=LogNorm(vmin=10 ** (-10), vmax=1.0),
                                  extent=[0, self.world_grid.total_width, 0, self.world_grid.total_height], zorder=0)
            belief.draw(artist_b0, item_nr=item_map[key], world_width=self.world_grid.total_width,
                        world_height=self.world_grid.total_height)
            fig.colorbar(artist_b0)
            # draw environment on top
            for line in self.lines_env:
                ax.plot(line[0], line[1], color='black', linewidth=2)
            # draw item itself
            if items_conf != -1:
                for item_line in items_conf:
                    if item_line.cb_type.get() != key:
                        continue
                    item_type = item_line.cb_type.get()
                    x, y = item_line.tb_x.get(), item_line.tb_y.get()
                    try:
                        x, y = float(x), float(y)
                    except:
                        x, y = 0, 0
                    if item_type == 'mug':
                        c = 'yellow'
                    elif item_type == 'plate':
                        c = 'green'
                    elif item_type == 'milk':
                        c = 'blue'
                    elif item_type == 'donut':
                        c = 'brown'
                    else:
                        print('invalid item type')
                    markersize = max(2, 5 - len(items_conf))
                    ax.plot(x, y, marker='s', markersize=markersize, color=c, label='x{}'.format(item_map[key]))

            plt.legend()
        plt.show()

    def sample_item_locations(self, nr_of_samples, nr_of_items, beliefspots_conf, beliefspot_width, items_conf,
                              scenario_name):
        # create rectangles for nodes
        self.initialize_grids()
        # initialize belief
        # map item names to item_nr
        item_map = {}
        for idx, item_line in enumerate(items_conf):
            item_map[item_line.cb_type.get()] = idx
        belief = Belief(total_nr_of_cells=10000, recs_beliefgrids=self.rec_env, nr_of_items=len(items_conf))
        # set belief spots
        for key in item_map:
            for b_line in beliefspots_conf:
                if b_line.cb_type.get() == key:
                    x, y, w = float(b_line.tb_x.get()), float(b_line.tb_y.get()), float(b_line.tb_prob.get())
                    belief.add_belief_spot(mu_x=x, mu_y=y, prob=w, sigma=beliefspot_width, item_nr=item_map[key])

        # create nr_of_samples random numbers
        all_items_x, all_items_y = np.zeros((nr_of_items, nr_of_samples)), np.zeros((nr_of_items, nr_of_samples))
        for sample_idx in range(nr_of_samples):
            for item_idx in range(nr_of_items):
                rnd_number_is_valid = False
                while not rnd_number_is_valid:
                    rnd_number = random.random()
                    b_aggregated = 0
                    for bg in belief.belief_grids:
                        b_aggregated += bg.get_aggregated_belief(item_nr=item_idx)
                        if b_aggregated > rnd_number:
                            b_aggregated -= bg.get_aggregated_belief(item_nr=item_idx)
                            break
                    # find exact belief_grid_cell
                    finished_loop = False
                    for v in range(len(bg.data[item_idx])):
                        for u in range(len(bg.data[item_idx, 0])):
                            b_aggregated += bg.data[item_idx, v, u]
                            if b_aggregated > rnd_number:
                                x, y = bg.get_xy(u, v)
                                # change x,y to middle of agent-grid cell
                                u_agent_grid, v_agent_grid = self.agent_grid.get_cell_indices_by_position(x, y)
                                x_agent_grid, y_agent_grid = self.agent_grid.get_position_by_indices(u_agent_grid,
                                                                                                     v_agent_grid)
                                # check distance to other items
                                for idx in range(len(all_items_x[:, sample_idx])):
                                    dist = (all_items_x[idx, sample_idx] - x_agent_grid) ** 2 + \
                                           (all_items_y[idx, sample_idx] - y_agent_grid) ** 2
                                    # if distance too closed to another item: start again
                                    if dist < 1.2 * ((3 * self.world_grid.cell_width) ** 2 +
                                                     (3 * self.world_grid.cell_height) ** 2):
                                        finished_loop = True
                                        break
                                # check if not on edge of world (leads to reachability problems)

                                if not self.agent_grid.cells[v_agent_grid][u_agent_grid].value == 'free':
                                    finished_loop = True
                                if finished_loop:
                                    break
                                else:
                                    all_items_x[item_idx, sample_idx] = x_agent_grid
                                    all_items_y[item_idx, sample_idx] = y_agent_grid
                                    rnd_number_is_valid = True
                                    finished_loop = True
                                    break
                        if finished_loop:
                            break
        # write item positions to txt file
        f = open("item_x_samples_" + scenario_name.replace(' ', '') + ".txt", "w+")
        for items_x in all_items_x:
            f.write(' '.join(map(str, items_x)) + '\n')
        f.close()
        f = open("item_y_samples_" + scenario_name.replace(' ', '') + ".txt", "w+")
        for items_y in all_items_y:
            f.write(' '.join(map(str, items_y)) + '\n')
        f.close()
