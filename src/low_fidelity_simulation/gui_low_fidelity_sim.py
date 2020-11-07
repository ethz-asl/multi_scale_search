import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import numpy as np

import config
from src.low_fidelity_simulation.low_fidelity_sim import LowFidelitySimulation
from src.multi_scale_search import auxiliary_functions


class GuiItemConf:
    def __init__(self, item_nr, row_nr, frame):
        self.item_nr = item_nr
        self.row_nr = row_nr
        self.lbl = tk.Label(master=frame, text='item' + str(self.item_nr) + ': item type:', font=('Arial', 12))
        self.lbl.grid(column=0, row=self.row_nr)
        self.cb_type = ttk.Combobox(master=frame, width=12)
        self.cb_type['values'] = ('mug', 'plate', 'milk', 'donut')
        self.cb_type.grid(column=1, row=self.row_nr)
        self.cb_type.current(0)
        self.lbl_in = tk.Label(master=frame, text=', initial x,y:', font=('Arial', 12))
        self.lbl_in.grid(column=2, row=self.row_nr, sticky='w')
        self.tb_x = tk.Entry(frame, width=5)
        self.tb_x.grid(column=3, row=self.row_nr)
        self.tb_y = tk.Entry(frame, width=5)
        self.tb_y.grid(column=4, row=self.row_nr)
        self.lbl_g = tk.Label(master=frame, text=', goal x,y:', font=('Arial', 12))
        self.lbl_g.grid(column=5, row=self.row_nr)
        self.tb_gx = tk.Entry(frame, width=5)
        self.tb_gx.grid(column=6, row=self.row_nr)
        self.tb_gy = tk.Entry(frame, width=5)
        self.tb_gy.grid(column=7, row=self.row_nr)

    def destroy(self):
        self.lbl.destroy()
        self.cb_type.destroy()
        self.lbl_in.destroy()
        self.tb_x.destroy()
        self.tb_y.destroy()
        self.lbl_g.destroy()
        self.tb_gx.destroy()
        self.tb_gy.destroy()


class GUIBeliefConf:
    def __init__(self, frame, row_nr):
        self.row_nr = row_nr
        self.lbl_blv = tk.Label(master=frame, text='Belief Spot: item type', font=('Arial', 12), anchor='w')
        self.lbl_blv.grid(column=0, row=row_nr, sticky='w')
        self.cb_type = ttk.Combobox(master=frame, width=6)
        self.cb_type['values'] = ('mug', 'plate', 'milk', 'donut')
        self.cb_type.grid(column=1, row=row_nr)
        self.cb_type.current(0)
        self.lbl_coord = tk.Label(master=frame, text=', coord (x,y):', font=('Arial', 12), anchor='w')
        self.lbl_coord.grid(column=2, row=row_nr, sticky='w')
        self.tb_x = tk.Entry(frame, width=5)
        self.tb_x.grid(column=3, row=row_nr)
        self.tb_y = tk.Entry(frame, width=5)
        self.tb_y.grid(column=4, row=row_nr)
        self.lbl_prob = ttk.Label(master=frame, text=', prob.:', font=('Arial', 12), anchor='w')
        self.lbl_prob.grid(column=5, row=row_nr)
        self.tb_prob = tk.Entry(frame, width=5)
        self.tb_prob.grid(column=6, row=row_nr)

    def destroy(self):
        self.lbl_blv.destroy()
        self.cb_type.destroy()
        self.cb_type.destroy()
        self.lbl_coord.destroy()
        self.tb_x.destroy()
        self.tb_y.destroy()
        self.lbl_prob.destroy()
        self.tb_prob.destroy()


# GUI For the Low Fidelity Simulation (LFS)
class GUI_LFS:
    def __init__(self):
        config.test_var = 7
        # initialize low_fidelity_simulation
        self.sim = LowFidelitySimulation()
        # create Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Simulation")
        self.root.geometry('1300x800')

        # create the Frames for the different sections of the GUI
        self.conf_frame = tk.Frame(self.root)  # conf = configuration
        self.conf_frame.pack(side=tk.TOP, anchor='w', expand=tk.NO)
        self.belief_frame = tk.Frame(self.root)
        self.belief_frame.pack(side=tk.TOP, anchor='w')
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, anchor='w')
        self.evaluation_frame = tk.Frame(self.root)
        self.evaluation_frame.pack(side=tk.TOP, anchor='w')
        # abbreviations used: lbl = label, cb = combobox, tb = textbox, btn = button, rad=radiobutton
        # abbreviations used: env = environment, it = iterations, ag = agent,
        self.lbl_env = tk.Label(master=self.conf_frame, text="environment:", font=('Arial', 12))
        self.lbl_env.grid(column=0, row=0, sticky='w')
        self.cb_env = ttk.Combobox(master=self.conf_frame, width=12)
        self.cb_env['values'] = ('small', 'big')
        self.cb_env.current(0)
        self.cb_env.grid(column=1, row=0)
        self.lbl_maxit = tk.Label(master=self.conf_frame, text='max iterations', font=('Arial', 12))
        self.lbl_maxit.grid(column=0, row=1, sticky='w')
        self.tb_maxit = tk.Entry(self.conf_frame, width=5)
        self.tb_maxit.insert(0, 1200)
        self.tb_maxit.grid(column=1, row=1, sticky='w')
        self.lbl_ag = tk.Label(master=self.conf_frame, text='agent:', font=('Arial', 12), anchor='w')
        self.lbl_ag.grid(column=0, row=2, sticky='w')
        self.cb_ag = ttk.Combobox(master=self.conf_frame, width=12)
        self.cb_ag['values'] = config.agent_types
        self.cb_ag.grid(column=1, row=2)
        self.cb_ag.current(0)
        self.lbl_ag_conf0 = tk.Label(master=self.conf_frame, text='initial x,y,theta:', font=('Arial', 12), anchor='w')
        self.lbl_ag_conf0.grid(column=2, row=2, sticky='w')
        self.tb_ag_x = tk.Entry(self.conf_frame, width=5)
        self.tb_ag_x.grid(column=3, row=2)
        self.tb_ag_x.insert(0, '1.0')
        self.tb_ag_y = tk.Entry(self.conf_frame, width=5)
        self.tb_ag_y.insert(0, '1.0')
        self.tb_ag_y.grid(column=4, row=2)
        self.tb_ag_theta = tk.Entry(self.conf_frame, width=5)
        self.tb_ag_theta.insert(0, '0')
        self.tb_ag_theta.grid(column=5, row=2, sticky='w')
        # items and belief lines
        self.items_conf = []
        self.lbl_beliefspot_width = tk.Label(master=self.belief_frame, text='belief spot width:', font=('Arial', 12),
                                             anchor='w')
        self.lbl_beliefspot_width.grid(column=0, row=0, sticky='w')
        self.tb_beliefspot_width = tk.Entry(self.belief_frame, width=5)
        self.tb_beliefspot_width.grid(column=1, row=0)
        self.tb_beliefspot_width.insert(0, '0.1')
        self.beliefspots_conf = []
        self.__initial_items_belief_configuration()
        # buttons
        self.btn_add_item = tk.Button(self.button_frame, text='add item', command=lambda: self.__btn_add_item_command())
        self.btn_add_item.grid(column=0, row=0, sticky='w', padx=10, pady=10)
        self.btn_del_item = tk.Button(self.button_frame, text='del item', command=lambda: self.__btn_del_item_command())
        self.btn_del_item.grid(column=1, row=0, sticky='w', padx=10, pady=10)
        self.btn_add_belief = tk.Button(self.button_frame, text='add belief',
                                        command=lambda: self.__btn_add_bel_command())
        self.btn_add_belief.grid(column=2, row=0, sticky='w', padx=10, pady=10)
        self.btn_del_belief = tk.Button(self.button_frame, text='del belief',
                                        command=lambda: self.__btn_del_bel_command())
        self.btn_del_belief.grid(column=3, row=0, sticky='w', padx=10, pady=10)
        self.btn_plot_world = tk.Button(self.button_frame, text='plot world',
                                        command=lambda: self.__btn_plot_world_command())
        self.btn_plot_world.grid(column=0, row=1, sticky='w', padx=10, pady=10)
        self.btn_plot_belief = tk.Button(self.button_frame, text='plot b0',
                                         command=lambda: self.__btn_plot_b0_command())
        self.btn_plot_belief.grid(column=1, row=1, sticky='w', padx=10, pady=10)
        # button to start the agent
        self.btn_start = tk.Button(self.button_frame, text='Start Simulation',
                                   command=lambda: self.__btn_start_command())
        self.btn_start.grid(column=2, row=1, sticky='w', padx=10, pady=10)
        self.save_video = tk.IntVar()
        self.chb_save_video = tk.Checkbutton(self.button_frame, variable=self.save_video, text='save video',
                                             onvalue=1, offvalue=0)
        self.chb_save_video.grid(column=3, row=1, sticky='w')

        # evaluation options
        self.lbl_visualisation = tk.Label(master=self.evaluation_frame, text='vis. resolution cells per meter:',
                                          font=('Arial', 12))
        self.lbl_visualisation.grid(column=0, row=0, sticky='w')
        self.tb_visualisation = tk.Entry(self.evaluation_frame, width=3)
        self.tb_visualisation.grid(column=1, row=0, sticky='w')
        self.tb_visualisation.insert(0, 1)
        self.lbl_timeout_time = tk.Label(master=self.evaluation_frame, text="sarsop timeout time:", font=('Arial', 12))
        self.lbl_timeout_time.grid(column=0, row=1, sticky='w')
        self.tb_timeout_time = tk.Entry(self.evaluation_frame, width=3)
        self.tb_timeout_time.grid(column=1, row=1, sticky='w')
        self.tb_timeout_time.insert(0, 5.0)
        self.lbl_b0_threshold = tk.Label(master=self.evaluation_frame, text='b0 treshold:', font=('Arial', 12))
        self.lbl_b0_threshold.grid(column=0, row=2, sticky='w')
        self.tb_b0_threshold = tk.Entry(self.evaluation_frame, width=6)
        self.tb_b0_threshold.grid(column=1, row=2, sticky='w')
        self.tb_b0_threshold.insert(0, 0.999)
        self.lbl_testscenario = tk.Label(master=self.evaluation_frame, text='test scenario:',
                                         font=('Arial', 12))
        self.lbl_testscenario.grid(column=0, row=3, sticky='w')
        self.cb_testscenario = ttk.Combobox(master=self.evaluation_frame, width=12)
        self.cb_testscenario['values'] = ('scenario 01', 'scenario 02', 'scenario 03', 'scenario 04', 'scenario 05',
                                          'corner case 01', 'corner case 02')
        self.cb_testscenario.grid(column=1, row=3)
        self.btn_testscenario = tk.Button(self.evaluation_frame, text='set test scenario',
                                          command=lambda: self.__btn_set_test_scenario())
        self.btn_testscenario.grid(column=2, row=3, sticky='w', padx=10, pady=10)

        # self.lbl_evaluation_mode_b = tk.Label(master=self.evaluation_frame, text='evaluation mode b:', font='Arial')
        # self.lbl_evaluation_mode_b.grid(column=0, row=4, sticky='w')
        # self.lbl_nr_of_samples = tk.Label(master=self.evaluation_frame, text='nr of samples:', font=('Arial', 12))
        # self.lbl_nr_of_samples.grid(column=1, row=4, sticky='w')
        # self.tb_nr_of_samples = tk.Entry(self.evaluation_frame, width=5)
        # self.tb_nr_of_samples.grid(column=2, row=4, sticky='w')
        # self.tb_nr_of_samples.insert(0, 10)
        # self.btn_create_samples = tk.Button(self.evaluation_frame, text='create samples',
        #                                     command=lambda: self.__btn_sample_command())
        # self.btn_create_samples.grid(column=3, row=4, sticky='w', padx=10, pady=10)
        # self.btn_start_evaluation = tk.Button(self.evaluation_frame, text='start evaluation',
        #                                       command=lambda: self.__btn_start_evaluation_command())
        # self.btn_start_evaluation.grid(column=4, row=4, sticky='w', padx=10, pady=10)
        # self.btn_save_conf = tk.Button(self.evaluation_frame, text='save configuration',
        #                                command=lambda: self.__btn_save_conf_command())
        # self.btn_save_conf.grid(column=0, row=5, sticky='w', padx=10, pady=10)

    def run(self):
        self.root.mainloop()

    # private methods follow #############
    def __run_simulation(self):
        # initialisation
        # convert configuration to the correct datatype and give error message if this is not the case
        try:
            agent_type, ag_conf0_x, ag_conf0_y, ag_conf0_theta, environment_type, max_it, visualisation_ratio, timeout_time, b0_threshold, beliefspots_width = self.__check_non_item_variables_format()
        except:
            return 'E:init'
        self.sim.set_agent_type(agent_type)
        self.sim.set_agent_conf0(ag_conf0_x, ag_conf0_y, ag_conf0_theta)
        self.sim.set_env_type(environment_type)
        self.sim.set_max_it(max_it)
        self.sim.set_visualisation_ratio(visualisation_ratio)
        self.sim.set_timeout_time(timeout_time)
        self.sim.set_b0_threshold(b0_threshold)
        self.sim.set_print_every_timestep(True)
        self.sim.set_save_video(self.save_video.get())
        # delete all current items
        self.sim.delete_items()
        item_map = {}
        item_nr = 0
        for item_line in self.items_conf:
            # check if item_type is legit
            if item_line.cb_type.get() not in config.item_types:
                messagebox.showinfo('Error', 'item_type is not valid')
                return 'init_error'
            # convert configuration to the correct datatype and give error message if this is not the case
            try:
                x, y = float(item_line.tb_x.get()), float(item_line.tb_y.get())
                g_x, g_y = float(item_line.tb_gx.get()), float(item_line.tb_gy.get())
            except (TypeError, ValueError) as e:
                messagebox.showinfo('Error', 'item configuration is not in correct format')
                print(e)
                return 'init_error'
            return_value = self.sim.set_item(item_line.cb_type.get(), x, y, g_x, g_y)
            if return_value == 'E:item':
                messagebox.showinfo('Error', 'item not in reachable space')
                return 'init_error'
            elif return_value == 'E:goal':
                messagebox.showinfo('Error', 'item-goal not in reachable space')
                return 'init_error'
            elif return_value == 'E:dist':
                messagebox.showinfo('Error', 'items are too close together')
                return 'init_error'
            # add item to item_map:
            item_map[item_line.cb_type.get()] = item_nr
            item_nr += 1
        # set belief
        # check belief for correct format
        if agent_type != 'b1':
            belief_spots = np.zeros((len(self.beliefspots_conf), 4))
            for i in range(len(self.beliefspots_conf)):
                if self.beliefspots_conf[i].cb_type.get() not in item_map.keys():
                    messagebox.showinfo('Error', 'item of belief spot is not set in items')
                    return 'init_error'
                # convert configuration to the correct datatype and give error message if this is not the case
                try:
                    x, y = float(self.beliefspots_conf[i].tb_x.get()), float(self.beliefspots_conf[i].tb_y.get())
                    prob = float(self.beliefspots_conf[i].tb_prob.get())
                except (TypeError, ValueError) as e:
                    messagebox.showinfo('Error', 'belief configuration is not in correct format')
                    print(e)
                    return 'init_error'
                belief_spots[i, :] = [x, y, prob, item_map[self.beliefspots_conf[i].cb_type.get()]]
            self.sim.set_belief(belief_spots, beliefspots_width)
        else:
            messagebox.showinfo('Error', 'something went wrong in MDP/POMDP selection')
            return 'init_error'
        self.sim.reinitialize()
        print('correctly initialized Simulation ################################################')
        # run simulation
        return_value = self.sim.run_sim()
        return return_value

    def __initial_items_belief_configuration(self):
        # item line
        item1_line = GuiItemConf(1, row_nr=3, frame=self.conf_frame)
        self.items_conf.append(item1_line)
        self.items_conf[0].cb_type.current(0)
        self.items_conf[0].tb_x.insert(0, '2')
        self.items_conf[0].tb_y.insert(0, '2')
        self.items_conf[0].tb_gx.insert(0, '2.0')
        self.items_conf[0].tb_gy.insert(0, '8.0')
        # belief lines
        self.beliefspots_conf.append(GUIBeliefConf(self.belief_frame, 1))
        # add 2 belief spots
        self.beliefspots_conf[0].tb_x.insert(0, '2')
        self.beliefspots_conf[0].tb_y.insert(0, '2')
        self.beliefspots_conf[0].tb_prob.insert(0, '0.8')

    def __btn_add_item_command(self):
        item_i = GuiItemConf(item_nr=self.items_conf[-1].item_nr + 1, row_nr=self.items_conf[-1].row_nr + 1,
                             frame=self.conf_frame)
        self.items_conf.append(item_i)

    def __btn_del_item_command(self):
        if len(self.items_conf) > 1:
            # destroy widgets
            item_line = self.items_conf.pop()
            item_line.destroy()

    def __btn_add_bel_command(self):
        row_nr = 0
        if len(self.beliefspots_conf) > 0:
            row_nr = self.beliefspots_conf[-1].row_nr
        bel_i = GUIBeliefConf(frame=self.belief_frame, row_nr=row_nr + 1)
        self.beliefspots_conf.append(bel_i)

    def __btn_del_bel_command(self):
        if len(self.beliefspots_conf) > 0:
            bel_line = self.beliefspots_conf.pop()
            bel_line.destroy()

    def __btn_plot_world_command(self):
        # initialize grid with current settings
        if self.cb_env.get() != 'small' and self.cb_env.get() != 'big':
            messagebox.showinfo('Error', 'invalid/undefined environment')
            return
        self.sim.set_env_type(self.cb_env.get())
        try:
            ag_conf0_x, ag_conf0_y = float(self.tb_ag_x.get()), float(self.tb_ag_y.get())
        except (TypeError, ValueError) as e:
            messagebox.showinfo('Error', 'agent configuration is not in correct format')
            print(e)
            return
        self.sim.set_agent_conf0(ag_conf0_x, ag_conf0_y, 0)
        self.sim.draw_world(self.items_conf)

    def __btn_plot_b0_command(self):
        if self.cb_env.get() != 'small' and self.cb_env.get() != 'big':
            messagebox.showinfo('Error', 'environment is not in correct format')
            return
        for i in range(len(self.beliefspots_conf)):
            # convert configuration to the correct datatype and give error message if this is not the case
            try:
                x, y = float(self.beliefspots_conf[i].tb_x.get()), float(self.beliefspots_conf[i].tb_y.get())
                prob = float(self.beliefspots_conf[i].tb_prob.get())
            except (TypeError, ValueError) as e:
                messagebox.showinfo('Error', 'belief configuration is not in correct format')
                print(e)
                return
        self.sim.set_env_type(self.cb_env.get())
        self.sim.draw_b0(self.beliefspots_conf, float(self.tb_beliefspot_width.get()), self.items_conf)

    def __btn_start_command(self):
        # run simulation
        return_value = self.__run_simulation()
        if return_value == 'E:init':
            pass
        elif return_value == 'init_error':
            pass
        else:
            self.sim.draw_documentation()

    def __btn_set_test_scenario(self):
        test_scenario = self.cb_testscenario.get()
        nr_of_items = 0
        ag_x, ag_y = 0.5, 0.5
        belief_spot_sigma = 0.1
        item_types, init_x, init_y, goal_x, goal_y, belief_types, b_x, b_y, b_p = [], [], [], [], [], [], [], [], []
        if test_scenario == 'scenario 01':
            self.cb_env.set('small')
            nr_of_items = 1
            item_types = ['mug']
            init_x = [2.0]
            init_y = [10.0]
            goal_x = [4.0]
            goal_y = [8.0]
            belief_types = ['mug', 'mug']
            b_x = [2.0, 5.0]
            b_y = [10.0, 2.0]
            b_p = [0.6, 0.3]

        elif test_scenario == 'scenario 02':
            self.cb_env.set('small')
            nr_of_items = 1
            item_types = ['mug']
            init_x = [8]
            init_y = [4.5]
            goal_x = [0.5]
            goal_y = [3]
            belief_types = ['mug', 'mug', 'mug']
            b_x = [13.0, 5.0, 1.0]
            b_y = [12.0, 3.5, 10.0]
            b_p = [0.3, 0.4, 0.2]

        elif test_scenario == 'scenario 03':
            self.cb_env.set('small')
            nr_of_items = 2
            item_types = ['mug', 'plate']
            init_x = [5.5, 0.5]
            init_y = [3.4, 10]
            goal_x = [4.5, 1.0]
            goal_y = [6.5, 1.0]
            belief_types = ['mug', 'mug', 'plate']
            b_x = [5.0, 10.0, 0.5]
            b_y = [3.4, 5.0, 10.0]
            b_p = [0.3, 0.6, 0.6]

        elif test_scenario == 'scenario 04':
            self.cb_env.set('small')
            nr_of_items = 2
            item_types = ['mug', 'plate']
            init_x = [0.5, 9.0]
            init_y = [10.0, 5.0]
            # init_x = [9.381578947368421, 9.267543859649123]
            # init_y = [12.875, 11.965909090909092]
            goal_x = [7.0, 1.0]
            goal_y = [8.0, 1.0]
            belief_types = ['mug', 'mug', 'mug', 'mug', 'plate', 'plate', 'plate', 'plate']
            b_x = [0.5, 0.5, 4.5, 4.5, 9.0, 13.0, 9.0, 13.0]
            b_y = [7.0, 10.0, 7.0, 10.0, 5.0, 5.0, 12.0, 12.0]
            b_p = [0.15] * 8

        elif test_scenario == 'scenario 05':
            self.cb_env.set('big')
            ag_x, ag_y = 1.0, 9.5
            belief_spot_sigma = 0.2
            nr_of_items = 1
            item_types = ['mug']
            init_x = [23.0]
            init_y = [6.0]
            goal_x = [24.0]
            goal_y = [19.0]
            belief_types = ['mug', 'mug']
            b_x = [10.0, 23.0]
            b_y = [23.0, 6.0]
            b_p = [0.3, 0.5]

        elif test_scenario == 'corner case 01':
            self.cb_env.set('big')
            ag_x, ag_y = 1.0, 9.5
            belief_spot_sigma = 0.2
            nr_of_items = 1
            item_types = ['mug']
            init_x = [10.6]
            init_y = [24.0]
            goal_x = [17.0]
            goal_y = [9.0]
            belief_types = ['mug', 'mug', 'mug']
            b_x = [10.0, 23.0, 3.0, 3.4]
            b_y = [23.0, 6.0, 15.0, 4.8]
            b_p = [0.2, 0.3, 0.2, 0.15]

        elif test_scenario == 'corner case 02':
            self.cb_env.set('big')
            ag_x, ag_y = 11.0, 10
            belief_spot_sigma = 0.05
            nr_of_items = 1
            item_types = ['mug']
            init_x = [13.5]
            init_y = [10.0]
            goal_x = [20.0]
            goal_y = [14.7]
            belief_types = ['mug']
            b_x = [13.5]
            b_y = [10.0]
            b_p = [0.8]
        else:
            messagebox.showinfo('Error', 'invalid test-scenario')
            return

        self.tb_ag_x.delete(0, tk.END)
        self.tb_ag_x.insert(0, str(ag_x))
        self.tb_ag_y.delete(0, tk.END)
        self.tb_ag_y.insert(0, str(ag_y))
        # delete or add item lines
        while len(self.items_conf) < nr_of_items:
            self.__btn_add_item_command()
        while len(self.items_conf) > nr_of_items:
            self.__btn_del_item_command()
        # set item values
        for idx, item_line in enumerate(self.items_conf):
            item_line.cb_type.set(item_types[idx])
            item_line.tb_x.delete(0, tk.END)
            item_line.tb_x.insert(0, str(init_x[idx]))
            item_line.tb_y.delete(0, tk.END)
            item_line.tb_y.insert(0, str(init_y[idx]))
            item_line.tb_gx.delete(0, tk.END)
            item_line.tb_gx.insert(0, str(goal_x[idx]))
            item_line.tb_gy.delete(0, tk.END)
            item_line.tb_gy.insert(0, str(goal_y[idx]))
        # delete or add belief lines
        while len(self.beliefspots_conf) < len(belief_types):
            self.__btn_add_bel_command()
        while len(self.beliefspots_conf) > len(belief_types):
            self.__btn_del_bel_command()
        # set belief values
        for idx, belief_line in enumerate(self.beliefspots_conf):
            belief_line.cb_type.set(belief_types[idx])
            belief_line.tb_x.delete(0, tk.END)
            belief_line.tb_x.insert(0, str(b_x[idx]))
            belief_line.tb_y.delete(0, tk.END)
            belief_line.tb_y.insert(0, str(b_y[idx]))
            belief_line.tb_prob.delete(0, tk.END)
            belief_line.tb_prob.insert(0, str(b_p[idx]))
        # set belief spot sigma
        self.tb_beliefspot_width.delete(0, tk.END)
        self.tb_beliefspot_width.insert(0, belief_spot_sigma)
        return

    def __btn_sample_command(self):
        self.sim.sample_item_locations(int(self.tb_nr_of_samples.get()), len(self.items_conf), self.beliefspots_conf,
                                       float(self.tb_beliefspot_width.get()), self.items_conf, self.cb_testscenario.get())

    def __btn_start_evaluation_command(self):
        # read item_x, item_y from file
        scenario_name = self.cb_testscenario.get()
        try:
            with open('item_x_samples_' + scenario_name.replace(' ', '') + '.txt') as f:
                item_x_str = [line.strip() for line in f]
        except:
            messagebox.showinfo('Need to create Samples first!')
            return
        item_x_samples = []
        for line in item_x_str:
            item_x_samples.append([float(val) for val in line.split(' ')])
        try:
            with open('item_y_samples_' + scenario_name.replace(' ', '') + '.txt') as f:
                item_y_str = [line.strip() for line in f]
        except:
            messagebox.showinfo('Need to create Samples first!')
        item_y_samples = []
        for line in item_y_str:
            item_y_samples.append([float(val) for val in line.split(' ')])
        if len(item_x_samples) != len(item_y_samples) or len(item_x_samples[0]) != len(item_y_samples[0]):
            messagebox.showinfo('Error', 'something wrong with item_samples, need to resample')
            return
        if len(item_x_samples) != len(self.items_conf):
            messagebox.showinfo('warning', 'item_samples do not correspond to current settings, need to resample?')
            return
        nr_of_samples = len(item_x_samples[0])
        items = []
        for item_line in self.items_conf:
            item_type = item_line.cb_type.get()
            item_gx = item_line.tb_gx.get()
            item_gy = item_line.tb_gy.get()
            items.append({'item_type': item_type, 'item_gx': item_gx, 'item_gy': item_gy})
        belief_spots = []
        for belief_line in self.beliefspots_conf:
            belief_spots.append({'belief_type': belief_line.cb_type.get(), 'belief_x': belief_line.tb_x.get(),
                                 'belief_y': belief_line.tb_y.get(), 'belief_p': belief_line.tb_prob.get()})
        # write simulation specifications to text tile
        self.__run_evaluation(nr_of_samples, items, belief_spots, item_x_samples, item_y_samples)

    def __btn_save_conf_command(self):
        environment_type = self.cb_env.get()
        try:
            ag_conf0_x, ag_conf0_y, ag_conf0_theta = float(self.tb_ag_x.get()), float(self.tb_ag_y.get()), \
                                                     auxiliary_functions.angle_consistency(
                                                         auxiliary_functions.deg_to_rad(float(self.tb_ag_theta.get())))
        except:
            messagebox.showinfo('Error', 'agent configuration is not in correct format')
            return 'init_error'
        if self.cb_ag.get() not in config.agent_types:
            messagebox.showinfo('Error', 'invalid agent type for this environment')
            return 'init_error'
        if environment_type != 'small' and environment_type != 'big':
            messagebox.showinfo('Error', 'environment is not in correct format')
            return 'init_error'
        try:
            max_it = int(self.tb_maxit.get())
        except:
            messagebox.showinfo('Error', 'invalid max_it format, needs to be integer')
            return 'init_error'
        try:
            timeout_time = float(self.tb_timeout_time.get())
        except:
            messagebox.showinfo('Error', 'timeout_time needs to be a float')
        try:
            b0_threshold = float(self.tb_b0_threshold.get())
            if b0_threshold > 1.0:
                b0_threshold = 1.0
            elif b0_threshold < 0.99:
                b0_threshold = 0.99
        except:
            messagebox.showinfo('Error', 'b0_threshold needs to be a float between 0.99 and 1.0 (reccomended: 0.999)')
        # delete all current items
        self.sim.delete_items()
        for item_line in self.items_conf:
            # check if item_type is legit
            if item_line.cb_type.get() not in config.item_types:
                messagebox.showinfo('Error', 'item_type is not valid')
                return 'init_error'
            # convert configuration to the correct datatype and give error message if this is not the case
            try:
                x, y = float(item_line.tb_x.get()), float(item_line.tb_y.get())
                g_x, g_y = float(item_line.tb_gx.get()), float(item_line.tb_gy.get())
            except:
                messagebox.showinfo('Error', 'item configuration is not in correct format')
                return 'init_error'
            return_value = self.sim.set_item(item_line.cb_type.get(), x, y, g_x, g_y)
            if return_value == 'E:item':
                messagebox.showinfo('Error', 'item not in reachable space')
                return 'init_error'
            elif return_value == 'E:goal':
                messagebox.showinfo('Error', 'item-goal not in reachable space')
                return 'init_error'
            elif return_value == 'E:dist':
                messagebox.showinfo('Error', 'items are too close together')
                return 'init_error'
        # check belief for correct format
        if self.cb_ag.get() != 'b1':
            for i in range(len(self.beliefspots_conf)):
                # convert configuration to the correct datatype and give error message if this is not the case
                try:
                    x, y = float(self.beliefspots_conf[i].tb_x.get()), float(self.beliefspots_conf[i].tb_y.get())
                    prob = float(self.beliefspots_conf[i].tb_prob.get())
                except:
                    messagebox.showinfo('Error', 'belief configuration is not in correct format')
                    return 'init_error'

        f = open('problem_config.txt', "w+")
        f.write('environment,{}\n'.format(environment_type))
        f.write('maxit,{}\n'.format(max_it))
        f.write('agent,{}\n'.format(self.cb_ag.get()))
        f.write('ag_x,{}\n'.format(ag_conf0_x))
        f.write('ag_y,{}\n'.format(ag_conf0_y))
        f.write('ag_theta,{}\n'.format(ag_conf0_theta))
        for item_line in self.items_conf:
            f.write('item,{},{},{},{},{}\n'.format(item_line.cb_type.get(), item_line.tb_x.get(), item_line.tb_y.get(),
                                                   item_line.tb_gx.get(), item_line.tb_gy.get()))
        f.write('sigma,{}\n'.format(self.tb_beliefspot_width.get()))
        for belief_line in self.beliefspots_conf:
            f.write('beliefspot,{},{},{},{}\n'.format(belief_line.cb_type.get(),
                                                      belief_line.tb_x.get(), belief_line.tb_y.get(),
                                                      belief_line.tb_prob.get()))
        f.write('timeout,{}\n'.format(self.tb_timeout_time.get()))
        f.write('b0threshold,{}\n'.format(self.tb_b0_threshold.get()))
        scenario_name = self.cb_testscenario.get()
        if self.cb_testscenario.get() == '':
            scenario_name = 'no_name'
        f.write('scenario,{}\n'.format(scenario_name.replace(' ', '')))
        f.write('####################\n')
        f.close()

    # TODO: WORK IN PROGRESS
    def __run_evaluation(self, nr_of_samples, items, belief_spots, item_x_samples, item_y_samples):
        try:
            agent_type, ag_conf0_x, ag_conf0_y, ag_conf0_theta, environment_type, max_it, visualisation_ratio, timeout_time, b0_threshold, beliefspot_width = self.__check_non_item_variables_format()
        except:
            return 'E:init'
        self.sim.set_agent_type(agent_type)
        self.sim.set_agent_conf0(ag_conf0_x, ag_conf0_y, ag_conf0_theta)
        self.sim.set_env_type(environment_type)
        self.sim.set_max_it(max_it)
        self.sim.set_visualisation_ratio(visualisation_ratio)
        self.sim.set_timeout_time(timeout_time)
        self.sim.set_b0_threshold(b0_threshold)

        # write simulation specifications to text tile
        directory = 'evaluations/' + config.simulation_name + '/' + self.cb_env.get() + '/' + self.cb_testscenario.get().replace(
            ' ', '') + '/' + agent_type + '/'
        try:
            os.mkdir(directory)
        except OSError as e:
            print(e)
            return 'E:mkdir failed'
        f_overview = open(directory + 'overview.txt', 'w+')
        f_overview.write('agent_type={}\n'.format(agent_type))
        f_overview.write('max_it={}\n'.format(max_it))
        f_overview.write('timeout_time={}\n'.format(timeout_time))
        f_overview.write('b0_threshold={}\n'.format(b0_threshold))
        f_overview.write('beliefspot_width={}\n'.format(beliefspot_width))
        f_overview.write('number of samples={}\n'.format(nr_of_samples))
        f_overview.write('agent_x={}. agent_y={}. agent_theta={}\n'.format(ag_conf0_x, ag_conf0_y, ag_conf0_theta))

        for item in items:
            f_overview.write(
                'item_type={}, goal_x={}, goal_y={}\n'.format(item['item_type'], item['item_gx'], item['item_gy']))
        for belief_spot in belief_spots:
            f_overview.write('belief spot: item_type={}, b_x={}, b_y={}, b_p={}\n'.format(
                belief_spot['belief_type'], belief_spot['belief_x'], belief_spot['belief_y'], belief_spot['belief_p']))

        sol_qualities = []
        nr_of_iterations = []
        comp_times, comp_times_avr, comp_times_layers, comp_times_layers_avr = [], [], [], []

        for n in range(nr_of_samples):
            print('###########################################################')
            print('START RUN {}'.format(n))
            print('###########################################################')
            # set items
            for idx, item in enumerate(items):
                item['item_x'] = item_x_samples[idx][n]
                item['item_y'] = item_y_samples[idx][n]
            try:
                return_value = 'E:init'
                # return_value = run_simulation_no_GUI(sim, b0_threshold, timeout_time, environment_type, is_MDP, max_it,
                #                                      agent_type, ag_conf0_x, ag_conf0_y, ag_conf0_theta, items,
                #                                      belief_spots,
                #                                      )
                if return_value == 'E:init':
                    print('need to initialize simulation first')
                elif return_value == 'init_error':
                    do_nothing = True
                else:
                    nr_of_iterations.append(return_value[0])
                    sol_qualities.append(return_value[1])
                    comp_times.append(return_value[2])
                    comp_times_avr.append(return_value[3])
                    comp_times_layers.append(return_value[4])
                    comp_times_layers_avr.append(return_value[5])
                    if n == 0:
                        f_overview.write(
                            'run_nr, nr_of_iterations, sol. quality, comp. time, comp. time per act., comp. times layers, comp. times layers per act.\n')
                    f_overview.write('{}, {}, {}, {}, {}, {}, {}\n'.format(
                        n, return_value[0], return_value[1], return_value[2], return_value[3], return_value[4],
                        return_value[5]))

                    comp_times_all_actions_all_layers = return_value[6]
                    f_detail = open(directory + '/run{}.txt'.format(n), 'w+')
                    for entry in comp_times_all_actions_all_layers:
                        f_detail.write('{}\n'.format(entry))
                    f_detail.close()

            except:
                print('ERROR IN RUN {}\n'.format(n))
        try:
            f_overview.write('average solution quality: {}\n'.format(np.mean(sol_qualities)))
            f_overview.write('average computation time: {}\n'.format(np.mean(comp_times)))
            f_overview.write('average computation time per action: {}\n'.format(np.mean(comp_times_avr)))
            for layer in range(len(comp_times_layers[0])):
                f_overview.write('average computation time l{}: {}\n'.format(layer + 1, np.mean(
                    np.array(comp_times_layers)[:, layer])))
            for layer in range(len(comp_times_layers[0])):
                f_overview.write('average computation time l{} pe raction: {}\n'.format(layer + 1, np.mean(
                    np.array(comp_times_layers_avr)[:, layer])))
        except:
            print('ERROR IN WRITING OVERVIEW FILE')
        f_overview.close()
        print('###########################################################')
        print('EVALUATION FINISHED')
        print('###########################################################')

    def __check_non_item_variables_format(self):
        try:
            ag_conf0_x, ag_conf0_y, ag_conf0_theta = float(self.tb_ag_x.get()), float(self.tb_ag_y.get()), \
                                                     auxiliary_functions.angle_consistency(
                                                         auxiliary_functions.deg_to_rad(float(self.tb_ag_theta.get())))
        except (TypeError, ValueError) as e:
            messagebox.showinfo('Error', 'agent configuration is not in correct format')
            print(e)
            return 'init_error'
        agent_type = self.cb_ag.get()
        environment_type = self.cb_env.get()
        if not agent_type in config.agent_types:
            messagebox.showinfo('Error', 'invalid agent type for this environment')
            return 'init_error'
        try:
            max_it = int(self.tb_maxit.get())
        except (TypeError, ValueError) as e:
            messagebox.showinfo('Error', 'invalid max_it format, needs to be integer')
            print(e)
            return 'init_error'
        try:
            visualisation_ratio = -1
            if self.tb_visualisation.get() != 'none':
                visualisation_ratio = int(self.tb_visualisation.get())
        except (TypeError, ValueError) as e:
            messagebox.showinfo(('Error', 'invalid visualisation number, needs to be integer between 1 and 10'))
            print(e)
        except:
            messagebox.showinfo(('Error', 'invalid visualisation number, needs to be integer between 1 and 10'))
        try:
            timeout_time = float(self.tb_timeout_time.get())
        except (TypeError, ValueError) as e:
            messagebox.showinfo('Error', 'timeout_time needs to be a float')
            print(e)
            return 'init_error'
        try:
            b0_threshold = float(self.tb_b0_threshold.get())
            if b0_threshold > 1.0:
                b0_threshold = 1.0
            elif b0_threshold < 0.99:
                b0_threshold = 0.99
        except (TypeError, ValueError) as e:
            messagebox.showinfo('Error', 'b0_threshold needs to be a float between 0.99 and 1.0 (reccomended: 0.999)')
            print(e)
            return 'init_error'
        try:
            beliefspot_width = float(self.tb_beliefspot_width.get())
        except (TypeError, ValueError) as e:
            messagebox.showinfo('Error', 'beliefspot_width needs to be a positive float')
            print(e)
            return 'init_error'

        return agent_type, ag_conf0_x, ag_conf0_y, ag_conf0_theta, environment_type, max_it, visualisation_ratio, \
               timeout_time, b0_threshold, beliefspot_width
