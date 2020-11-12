from low_fidelity_simulation import config_low_fidelity_sim
import math
import os
# global variables

ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/../'
SARSOP_SRC_FOLDER = ROOT_FOLDER + 'third_party/sarsop/src/'
# initialize all variables with the default values which correspond to the low fidelity simulation.
# the script run_experiment.py overwrites all variables for the correct simulation environment
simulation_name = 'low_fidelity_simulation'
BASE_FOLDER_SIM = ''
solving_precision = 0.01
# note: AP corresponds to Action Propagation, AVP to Action-Value-Propagation, VP to Value-Propagation
agent_types = ['FLAT', 'MultiScaleAP', 'MultiScaleAVP', 'MultiScaleVP']
POMDP_agent_types = ['FLAT', 'MultiScaleAP', 'MultiScaleAVP', 'MultiScaleVP']
multiscale_agent_types = ['MultiScaleAP', 'MultiScaleAVP', 'MultiScaleVP']
robot_range = 1.0
item_types = ['mug', 'plate', 'milk', 'donut']
grid_ref_width = 14.0
grid_ref_height = 13.4
robot_speed = 1.  # in meter/seconds
robot_angular_speed = float(math.pi / 4.)  # in rad/seconds, = 45 degree/seconds
robot_release_time = 3.0
robot_pickup_time = 3.0
world_discretisation = 0.01
# to get robot_viewing_cone run create_viewing_cone.py and copy list from console output
robot_viewing_cone = [[(0, -0.200001), (2.59807707737872, -1.7000015)],
                      [(0, -0.18666760000000002), (2.648843661524374, -1.5950827578292353)],
                      [(0, -0.17333420000000002), (2.6963830376915476, -1.4884480787383791)],
                      [(0, -0.1600008), (2.74063728647326, -1.3802111359640434)],
                      [(0, -0.1466674), (2.781552490884217, -1.2704875548543297)],
                      [(0, -0.133334), (2.8190788020503463, -1.1593947719971496)],
                      [(0, -0.1200006), (2.853170499941977, -1.0470518921418366)],
                      [(0, -0.10666720000000002), (2.8837860490766527, -0.9335795430883531)],
                      [(0, -0.09333380000000001), (2.910888149123716, -0.8190997287208986)],
                      [(0, -0.0800004), (2.9344437803490178, -0.7037356803649687)],
                      [(0, -0.066667), (2.954424243844377, -0.5876117066489688)],
                      [(0, -0.05333360000000001), (2.97080519649278, -0.47085304205329725)],
                      [(0, -0.040000199999999986), (2.9835666806267156, -0.3535856943314236)],
                      [(0, -0.02666679999999999), (2.992693148343523, -0.2359362909888496)],
                      [(0, -0.013333400000000023), (2.9981734804481146, -0.11803192500699962)],
                      [(0, 0.0), (3.000001, 0.0)],
                      [(0, 0.013333399999999995), (2.9981734804481146, 0.11803192500699976)],
                      [(0, 0.02666679999999999), (2.992693148343523, 0.2359362909888496)],
                      [(0, 0.040000200000000014), (2.9835666806267156, 0.35358569433142384)],
                      [(0, 0.05333360000000001), (2.97080519649278, 0.47085304205329725)],
                      [(0, 0.066667), (2.954424243844377, 0.5876117066489684)],
                      [(0, 0.0800004), (2.9344437803490178, 0.703735680364969)],
                      [(0, 0.0933338), (2.910888149123716, 0.8190997287208989)],
                      [(0, 0.10666719999999999), (2.8837860490766527, 0.9335795430883531)],
                      [(0, 0.12000060000000004), (2.853170499941977, 1.0470518921418366)],
                      [(0, 0.13333399999999998), (2.819078802050346, 1.1593947719971498)],
                      [(0, 0.14666740000000003), (2.781552490884217, 1.2704875548543297)],
                      [(0, 0.16000080000000003), (2.7406372864732607, 1.3802111359640434)],
                      [(0, 0.17333419999999997), (2.6963830376915476, 1.4884480787383791)],
                      [(0, 0.18666760000000002), (2.648843661524374, 1.595082757829235)]]


def init(simulation_type_):
    if simulation_type_ == 'low_fidelity_simulation':
        global simulation_name, BASE_FOLDER_SIM, solving_precision, agent_types, POMDP_agent_types, \
            multiscale_agent_types, item_types, grid_ref_width, grid_ref_height, \
            robot_range, robot_speed, robot_angular_speed, robot_release_time, robot_pickup_time, world_discretisation

        simulation_name = config_low_fidelity_sim.simulation_name
        BASE_FOLDER_SIM = config_low_fidelity_sim.BASE_FOLDER_SIM
        solving_precision = config_low_fidelity_sim.solving_precision
        agent_types = config_low_fidelity_sim.agent_types
        POMDP_agent_types = config_low_fidelity_sim.POMDP_agent_types
        multiscale_agent_types = config_low_fidelity_sim.multiscale_agent_types
        robot_range = config_low_fidelity_sim.robot_range
        item_types = config_low_fidelity_sim.item_types
        grid_ref_width = config_low_fidelity_sim.grid_ref_width
        grid_ref_height = config_low_fidelity_sim.grid_ref_height
        robot_speed = config_low_fidelity_sim.robot_speed
        robot_angular_speed = config_low_fidelity_sim.robot_angular_speed
        robot_release_time = config_low_fidelity_sim.robot_release_time
        robot_pickup_time = config_low_fidelity_sim.robot_pickup_time
        world_discretisation = config_low_fidelity_sim.world_discretisation

    elif simulation_type_ == 'ros_simulation':
        # not implemented (yet)
        pass


def get_pickup_reward(environment):
    if simulation_name == config_low_fidelity_sim.simulation_name:
        return config_low_fidelity_sim.get_pickup_reward(environment)


def get_delivery_reward(environment):
    if simulation_name == config_low_fidelity_sim.simulation_name:
        return config_low_fidelity_sim.get_delivery_reward(environment)


def get_penalty_reward(environment, layer):
    if simulation_name == config_low_fidelity_sim.simulation_name:
        return config_low_fidelity_sim.get_penalty_reward(environment, layer)


def get_recs_of_nodes(rec_env, environment='small', nr_of_layers=2):
    if simulation_name == config_low_fidelity_sim.simulation_name:
        return config_low_fidelity_sim.get_recs_of_nodes(rec_env, environment, nr_of_layers)


def construct_nodegraph_multiscale(nodegraph, layer, environment='small'):
    if simulation_name == config_low_fidelity_sim.simulation_name:
        return config_low_fidelity_sim.construct_nodegraph_multiscale(nodegraph, layer, environment)
