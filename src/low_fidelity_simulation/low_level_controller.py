from src.auxiliary_files import path_planning
from src.multi_scale_search import auxiliary_functions

import math

# Controller for the Low Fidelity Simulation (LFS)
class ControllerLFS(object):
    def __init__(self, grid):
        self.target_state = 'none'
        self.reference = []
        self.grid = grid

    def get_control_input(self, agent_x, agent_y, agent_theta, target_state):
        # target_state is either a string or a tuple (goal_x, goal_y, goal_theta)
        if type(target_state) == str:
            return target_state
        elif len(target_state) == 3:
            if self.target_state != target_state:
                goal_x, goal_y, goal_theta = target_state[0], target_state[1], target_state[2]
                self.__transform_goalpose_to_reference(agent_x, agent_y, agent_theta, goal_x, goal_y, goal_theta)
                self.target_state = target_state
        else:
            print('Error in low_level_controller: target_state has wrong data type')
        if len(self.reference) > 0:
            ref_x, ref_y, ref_theta = self.reference.pop(0)
        else:
            # sometimes the goal_ref is just to remain in the same position in order to collect more observations
            ref_x, ref_y, ref_theta = agent_x, agent_y, agent_theta

        return ref_x, ref_y, ref_theta

    def __transform_goalpose_to_reference(self, agent_x, agent_y, agent_theta, goal_x, goal_y, goal_theta='none'):
        agent_u, agent_v = self.grid.get_cell_indices_by_position(agent_x, agent_y)
        goal_u, goal_v = self.grid.get_cell_indices_by_position(goal_x, goal_y)

        path, ex_time = path_planning.astar(start_u=agent_u, start_v=agent_v, theta0=agent_theta,
                                            goal_u=goal_u, goal_v=goal_v, grid=self.grid)
        reference = []
        for p in range(0, len(path)):
            p_u, p_v = path[p][0], path[p][1]
            p_x, p_y = self.grid.get_position_by_indices(p_u, p_v)
            # if in correct cell: get next point
            if agent_u == p_u and agent_v == p_v:
                continue
            theta = auxiliary_functions.angle_consistency(2 * math.pi - math.atan2(p_v - agent_v, p_u - agent_u))
            dtheta = auxiliary_functions.angle_consistency(theta - agent_theta)
            if not (-0.00001 < dtheta < 0.00001):
                if len(reference) > 0:
                    reference.append((reference[-1][0], reference[-1][1], theta))
                else:
                    reference.append((agent_x, agent_y, theta))
            reference.append((p_x, p_y, theta))
            agent_u, agent_v, agent_theta = p_u, p_v, theta
        if agent_theta != goal_theta and goal_theta != 'none':
            reference.append((reference[-1][0], reference[-1][1], goal_theta))

        self.reference = reference
        return self.reference
