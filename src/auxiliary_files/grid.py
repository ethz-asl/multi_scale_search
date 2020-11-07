import math
import logging

import numpy as np


class Grid:

    def __init__(self, nr_of_cells_x=-1, nr_of_cells_y=-1, world_width=-1, world_height=-1, default_value='free',
                 grid_name='none', default_seen=1, x0=0, y0=0):
        self.log = logging.getLogger(__name__)
        self.nr_of_cells_x = nr_of_cells_x
        self.nr_of_cells_y = nr_of_cells_y
        self.total_width = float(world_width)
        self.total_height = float(world_height)
        self.cell_width = float(world_width) / nr_of_cells_x
        self.cell_height = float(world_height) / nr_of_cells_y
        self.x0 = x0
        self.y0 = y0
        # create grid
        self.cells = []
        for v in range(0, self.nr_of_cells_y):
            row = []
            for u in range(0, self.nr_of_cells_x):
                cell_xy = Cell(value=default_value, seen=default_seen)
                row.append(cell_xy)
            self.cells.append(row)
        self.grid_name = grid_name

    # only the cells that are entirely in the rectangle take the value
    def set_region(self, rectangle, value, x0_beh='smaller', x1_beh='smaller', y0_beh='smaller', y1_beh='smaller'):
        # get edge point, only cells completely in rectangle are considered
        if x0_beh == 'smaller':
            u0 = math.ceil((rectangle.x0 - self.x0) / self.total_width * self.nr_of_cells_x)
        elif x0_beh == 'bigger':
            u0 = math.floor((rectangle.x0 - self.x0) / self.total_width * self.nr_of_cells_x)
        if y0_beh == 'smaller':
            v0 = math.ceil((rectangle.y0 - self.y0) / self.total_height * self.nr_of_cells_y)
        elif y0_beh == 'bigger':
            v0 = math.floor((rectangle.y0 - self.y0) / self.total_height * self.nr_of_cells_y)
        if x1_beh == 'smaller':
            u1 = math.floor((rectangle.x0 - self.x0 + rectangle.width) / self.total_width * self.nr_of_cells_x)
        elif x1_beh == 'bigger':
            u1 = math.ceil((rectangle.x0 - self.x0 + rectangle.width) / self.total_width * self.nr_of_cells_x)
        if y1_beh == 'smaller':
            v1 = math.floor((rectangle.y0 - self.y0 + rectangle.height) / self.total_height * self.nr_of_cells_y)
        elif y1_beh == 'bigger':
            v1 = math.ceil((rectangle.y0 - self.y0 + rectangle.height) / self.total_height * self.nr_of_cells_y)
        u0, u1, v0, v1 = int(u0), int(u1), int(v0), int(v1)
        # fill the cells in the rectangle with value
        for v in range(v0, v1):
            for u in range(u0, u1):
                self.cells[v][u].value = value

    def set_all_seen_values(self, value):
        for v in range(len(self.cells)):
            for u in range(len(self.cells[0])):
                self.cells[v][u].seen = value

    def get_cell_indices_by_position(self, x, y, round_up=False):
        u = (x - self.x0) / self.total_width * self.nr_of_cells_x
        v = (y - self.y0) / self.total_height * self.nr_of_cells_y
        if 0 <= u < self.nr_of_cells_x and 0 <= v < self.nr_of_cells_y:
            if not round_up:
                return int(u), int(v)
            else:
                return int(math.ceil(u)), int(math.ceil(v))
        else:
            return -1, -1

    def get_uv_by_position_numpy(self, x_array, y_array, round_up=False):
        u_array = (x_array - self.x0) / self.total_width * self.nr_of_cells_x
        v_array = (y_array - self.y0) / self.total_height * self.nr_of_cells_y
        selection_u = np.logical_and(0 <= u_array, u_array < self.nr_of_cells_x)
        selection_v = np.logical_and(0 <= v_array, v_array < self.nr_of_cells_y)
        selection = np.logical_and(selection_u, selection_v)
        if not round_up:
            u_array = u_array.astype(int)
            v_array = v_array.astype(int)
        else:
            u_array = np.ceil(u_array).astype(int)
            v_array = np.ceil(v_array).astype(int)
        u_array[np.logical_not(selection)] = -1
        v_array[np.logical_not(selection)] = -1

        return u_array, v_array

    # returns a cell-object found through the position
    def get_cell_by_position(self, x, y):
        u = int((x - self.x0) / self.total_width * self.nr_of_cells_x)
        v = int((y - self.y0) / self.total_height * self.nr_of_cells_y)
        if 0 <= u < self.nr_of_cells_x and 0 <= v < self.nr_of_cells_y:
            return self.cells[v][u]
        else:
            print('position outside of grid')
            self.log.warning('position outside of grid')
            return -1

    def get_position_by_indices(self, u, v):
        x = self.x0 + float(u) / self.nr_of_cells_x * self.total_width + self.cell_width / 2.
        y = self.y0 + float(v) / self.nr_of_cells_y * self.total_height + self.cell_height / 2.
        return x, y

    def update_grid(self, observations):
        # convert observations to numpy array
        observations_np = np.array(observations)
        obs_x, obs_y = observations_np[:, 0].astype(float), observations_np[:, 1].astype(float)
        # convert x,y to u,v values
        obs_u, obs_v = self.get_uv_by_position_numpy(x_array=obs_x, y_array=obs_y)
        # first update all cells with an observation 'free'
        sel = observations_np[:, 2] == 'free'
        obs_u_sel, obs_v_sel = obs_u[sel], obs_v[sel]
        for idx in range(len(obs_u_sel)):
            self.cells[obs_v_sel[idx]][obs_u_sel[idx]].value = 'free'
            self.cells[obs_v_sel[idx]][obs_u_sel[idx]].seen = 0
        # now update all cells with an observation 'occupied'
        sel = observations_np[:, 2] == 'occupied'
        obs_u_sel, obs_v_sel = obs_u[sel], obs_v[sel]
        for idx in range(len(obs_u_sel)):
            self.cells[obs_v_sel[idx]][obs_u_sel[idx]].value = observations_np[sel, 2][idx]
            self.cells[obs_v_sel[idx]][obs_u_sel[idx]].seen = 0
        # now update all cells with an observation of an item
        sel = np.logical_and(observations_np[:, 2] != 'free', observations_np[:, 2] != 'occupied')
        obs_u_sel, obs_v_sel = obs_u[sel], obs_v[sel]
        for idx in range(len(obs_u_sel)):
            self.cells[obs_v_sel[idx]][obs_u_sel[idx]].value = 'occupied'
            self.cells[obs_v_sel[idx]][obs_u_sel[idx]].seen = 0

    def set_cell_value(self, value, cell_u, cell_v):
        self.cells[cell_v][cell_u].set_value(value)

    def get_seen_as_2Dlist(self):
        return_list = []
        for v in range(len(self.cells)):
            row = []
            for u in range(len(self.cells[0])):
                row.append(self.cells[v][u].seen)
            return_list.append(row.copy())
        return return_list

    def get_nr_of_free_cells(self):
        return sum(
            [self.cells[v][u].value == 'free' for v in range(len(self.cells)) for u in range(len(self.cells[0]))])

    # returns a cell-object found through index
    def get_cell(self, u, v):
        return self.cells[v][u]

    def get_cell_value_by_index(self, u, v):
        if 0 <= u < len(self.cells[0]) and 0 <= v < len(self.cells):
            return self.cells[v][u].value
        else:
            return 'occupied'

    def get_cell_value_by_pos(self, x, y):
        cell = self.get_cell_by_position(x, y)
        return cell.value

    def get_valid_intersection_area(self, rec):
        intersection_area = 0
        cell_area = self.cell_width * self.cell_height
        u0, v0 = self.get_cell_indices_by_position(rec.x0, rec.y0)
        u1, v1 = self.get_cell_indices_by_position(rec.x0 + rec.width, rec.y0 + rec.height, round_up=True)
        for v in range(v0, v1 + 1):
            for u in range(u0, u0 + 1):
                if self.cells[v][u].value != 'occupied':
                    intersection_area += cell_area
        return intersection_area

    def is_cell_in_grid(self, u, v):
        if 0 <= v < len(self.cells) and 0 <= u < len(self.cells[0]):
            return True
        else:
            return False


class Cell:
    # note: index is rows first, then columns
    def __init__(self, value, seen=1):
        self.value = value  # for agent_b1 \in {free, observed}
        self.seen = seen  # 1 = unobserved, 0 = just observed

    def set_value(self, value):
        self.value = value
