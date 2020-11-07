from __future__ import division  # required for standard float division in python 2.7

import copy
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


class BeliefGrid:

    # nr_of_items decides how many values a cell in the grid has (1 belief value per item)
    def __init__(self, nr_of_cells_x, nr_of_cells_y, nr_of_items, rec, init_val=0):
        self.nr_of_cells_x = int(nr_of_cells_x)
        self.nr_of_cells_y = int(nr_of_cells_y)
        self.nr_of_items = int(nr_of_items)
        self.data = init_val * np.ones((self.nr_of_items, self.nr_of_cells_y, self.nr_of_cells_x))
        self.rec = rec
        self.cell_width = self.rec.width / self.nr_of_cells_x
        self.cell_height = self.rec.height / self.nr_of_cells_y
        self.orr = 0.99  # orr=object-recognition-rate
        self.orr2 = 0.99  # probability that object-recognition algorithm correctly identifies that an item is not present

    # data is either a list (1D-3D) or a numpy array (1D-3D)
    def fill_grid(self, data, item_nr=0):
        if len(np.shape(data)) == 1:
            for i in range(np.shape(data)[0]):
                self.data[item_nr, int(i / self.nr_of_cells_x), i % self.nr_of_cells_x] = data[i]
        elif len(np.shape(data)) == 2:
            for v in range(np.shape(data)[0]):
                for u in range(np.shape(data)[1]):
                    if type(data) == list:
                        self.data[item_nr, v, u] = data[v][u]
                    else:
                        self.data[item_nr, v, u] = data[v, u]
        elif len(np.shape(data)) == 3:
            for it_nr in range(np.shape(data)[0]):
                for v in range(np.shape(data)[1]):
                    for u in range(np.shape(data)[2]):
                        if type(data) == list:
                            self.data[it_nr, v, u] = data[it_nr][v][u]
                        else:
                            self.data[it_nr, v, u] = data[it_nr, v, u]
        else:
            print('BeliefGrid: data has wrong format')

    def normalize_grid(self, total_belief_pre, total_belief_post=1, item_nr=0):
        if total_belief_pre == 0:
            self.data[item_nr, :, :] *= 0
        else:
            try:
                self.data[item_nr, :, :] *= (total_belief_post / total_belief_pre)
            except RuntimeWarning:
                print('total_belief_post = {}, total_belief_pre = {}'.format(total_belief_post, total_belief_pre))

    def get_belief_value(self, x, y, item_nr=0):
        u = int((x - self.rec.x0) / self.rec.width * self.nr_of_cells_x)
        v = int((y - self.rec.y0) / self.rec.height * self.nr_of_cells_y)
        return self.data[item_nr, v, u]

    def get_belief_vector(self, u, v):
        return self.data[:, v, u]

    def get_aggregated_belief(self, item_nr=0):
        return np.sum(self.data[item_nr, :, :])

    def set_belief_point(self, x, y, weight, item_nr=0):
        u, v = self.get_uv(x, y)
        self.data[item_nr, v, u] = weight

    # TODO: implement case 2 where after some time everything is back to initialization belief
    def update_belief(self, obs_x, obs_y, obs_val, item_nr, is_item_in_obs):
        obs_u, obs_v = self.get_uv_numpy(x_array=obs_x, y_array=obs_y)
        # create a copy of the data that is updated
        data_copy = np.copy(self.data[item_nr])
        if not is_item_in_obs:
            # first update the observed cell values
            data_copy[obs_v, obs_u] = np.maximum((1 - self.orr) * self.data[item_nr, obs_v, obs_u], 0.0000000001)
            # then update the not-observed cell values
            self.data[item_nr] = np.maximum(self.orr2 * self.data[item_nr], 0.0000000001)
            # overwrite the observed values
            self.data[item_nr, obs_v, obs_u] = data_copy[obs_v, obs_u]
        if is_item_in_obs:
            # update specific value where item_obs is
            sel = obs_val[:] == item_nr
            data_copy[obs_v[sel], obs_u[sel]] = np.maximum(self.orr * self.data[item_nr, obs_v[sel], obs_u[sel]],
                                                           0.0000000001)
            # update all values
            self.data[item_nr] = np.maximum((1 - self.orr) * self.data[item_nr], 0.0000000001)
            # overwrite values from copy
            self.data[item_nr, obs_v[sel], obs_u[sel]] = data_copy[obs_v[sel], obs_u[sel]]

    def set_region_uniformly(self, rec, prob_value, item_nr):
        # get intersection rectangle
        rec_in = self.rec.get_intersection_rec(rec)
        if rec_in != 'no':
            prob_value /= rec.get_area() / rec_in.get_area()
            u0_in, v0_in = self.get_uv(rec_in.x0, rec_in.y0)
            u1_in, v1_in = self.get_uv(rec_in.x0 + rec_in.width, rec_in.y0 + rec_in.height)
            self.data[item_nr, v0_in:v1_in + 1, u0_in:u1_in + 1] = prob_value / (
                    (u1_in - u0_in + 1) * (v1_in - v0_in + 1))

    def get_uv(self, x, y):
        u = int((x - self.rec.x0) / self.rec.width * self.nr_of_cells_x)
        v = int((y - self.rec.y0) / self.rec.height * self.nr_of_cells_y)
        return u, v

    def get_uv_numpy(self, x_array, y_array):
        u_array = ((x_array - self.rec.x0) / self.rec.width * self.nr_of_cells_x).astype(int)
        v_array = ((y_array - self.rec.y0) / self.rec.height * self.nr_of_cells_y).astype(int)
        return u_array, v_array

    def get_xy(self, u, v):
        x = self.rec.x0 + (u + 0.5) / self.nr_of_cells_x * self.rec.width
        y = self.rec.y0 + (v + 0.5) / self.nr_of_cells_y * self.rec.height
        return x, y


class BeliefSpot:
    def __init__(self, mu_x=-1.0, mu_y=-1.0, prob=-1.0, sigma=-1.0, item_nr=-1):
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.prob = prob
        self.sigma = sigma
        self.item_nr = item_nr


class Belief:
    def __init__(self, total_nr_of_cells, recs_beliefgrids, nr_of_items):
        self.nr_of_items = nr_of_items
        self.b_tot = [1.0] * nr_of_items  # may later be changed to > 1.0 to handle multiple items of the same kind
        self.belief_spots = []
        total_area = 0
        for rec in recs_beliefgrids:
            total_area += rec.get_area()
        # create BeliefGrids
        self.belief_grids = []
        for n in range(len(recs_beliefgrids)):
            nr_of_cells = total_nr_of_cells * (recs_beliefgrids[n].get_area() / total_area)
            nr_of_cells_x = round(math.sqrt(recs_beliefgrids[n].width / recs_beliefgrids[n].height * nr_of_cells))
            nr_of_cells_y = round(math.sqrt(recs_beliefgrids[n].height / recs_beliefgrids[n].width * nr_of_cells))
            self.belief_grids += [BeliefGrid(nr_of_cells_x, nr_of_cells_y, nr_of_items, rec=recs_beliefgrids[n],
                                             init_val=1)]

        # fill belief uniformly
        self.normalize_belief(item_nr=-1)
        self.total_nr_of_cells = sum([bg.nr_of_cells_x * bg.nr_of_cells_y for bg in self.belief_grids])

    # if item_nr = -1 -> grid is filled for all items
    def fill_grid(self, type='uniform', item_nr=-1):
        if type == 'uniform':
            for bg in self.belief_grids:
                data = [1] * (bg.nr_of_cells_x * bg.nr_of_cells_y)
                if item_nr == -1:
                    for i in range(self.nr_of_items):
                        bg.fill_grid(data, item_nr=i)
                else:
                    bg.fill_grid(data, item_nr=item_nr)
            self.normalize_belief(item_nr=item_nr)
        else:
            print('type {} is not implemented yet'.format(type))

    def add_item(self):
        self.nr_of_items += 1
        self.b_tot.append(1.0)
        for bg in self.belief_grids:
            new_data = np.ones((1, bg.nr_of_cells_y, bg.nr_of_cells_x))
            # print('bg.data.shape = {}'.format(bg.data.shape))
            bg.data = np.concatenate((bg.data, new_data), axis=0)
            # print('new bg.data.shape = {}'.format(bg.data.shape))
        # normalize new belief variable
        self.normalize_belief(item_nr=self.nr_of_items - 1)

    def set_b_tot(self, item_nr, total_belief):
        self.b_tot[item_nr] = total_belief

    # observations = list with entry = tuple (x, y, value), value is either number of item or 'no'
    def update_belief(self, observations):
        if len(observations) == 0:
            print('NO OBSERVATIONS')
            return
        # convert observations in appropriate format
        observations_np = np.array(observations)
        obs_x, obs_y = observations_np[:, 0].astype(float), observations_np[:, 1].astype(float)
        # convert values to item_nr as float or -1 for 'free' or 'occupied' values
        obs_val = -1 * np.ones((len(obs_x)), dtype=int)
        for item_nr in range(self.nr_of_items):
            obs_val[observations_np[:, 2] == '{}'.format(item_nr)] = item_nr
        is_item_in_obs = [False] * self.nr_of_items
        for item_nr in range(self.nr_of_items):
            is_item_in_obs[item_nr] = item_nr in obs_val
        belief_grids = copy.deepcopy(self.belief_grids)
        # print([self.belief_grids[i].get_aggregated_belief(item_nr=0) for i in range(len(self.belief_grids))])
        for bg in belief_grids:
            # get all observations in bg and update belief grid of bg
            sel = bg.rec.are_points_in_rec(x_array=obs_x, y_array=obs_y)
            # update belief seperately for every item
            for item_nr in range(self.nr_of_items):
                # is_item_in_obs = item_nr in obs_val
                bg.update_belief(obs_x[sel], obs_y[sel], obs_val[sel], item_nr, is_item_in_obs=is_item_in_obs[item_nr])
                # remove the already processed observed points
            obs_x = obs_x[np.logical_not(sel)]
            obs_y = obs_y[np.logical_not(sel)]
            obs_val = obs_val[np.logical_not(sel)]
        # normalize entire belief grid
        self.normalize_belief(item_nr=-1, belief_grids=belief_grids)
        # print([self.belief_grids[i].get_aggregated_belief(item_nr=0) for i in range(len(self.belief_grids))])

    # warning: this function reinitializes the belief with all the belief spots
    def add_belief_spot(self, mu_x, mu_y, prob, sigma, item_nr):
        if item_nr >= self.nr_of_items:
            print('you first need to add the item to the agent before you can set the belief')
            return 'warning'
        self.belief_spots.append(BeliefSpot(mu_x, mu_y, prob, sigma, int(item_nr)))
        self.set_all_belief_spots()

    def delete_belief_spots(self, item_nr):
        belief_spots_new = []
        for idx, bs in enumerate(self.belief_spots):
            if bs.item_nr != item_nr:
                belief_spots_new.append(bs)
        self.belief_spots = belief_spots_new

    def set_all_belief_spots(self):
        # fill the grid uniformly
        for item_nr in range(self.nr_of_items):
            b_sum = sum([bs.prob for bs in self.belief_spots if bs.item_nr == item_nr])
            self.set_b_tot(item_nr, total_belief=1.0 - b_sum)
            self.fill_grid(type='uniform', item_nr=item_nr)
        # fill belief spots
        for belief_spot in self.belief_spots:
            mu_x, mu_y = belief_spot.mu_x, belief_spot.mu_y,
            prob, sigma = belief_spot.prob, belief_spot.sigma
            item_nr = int(belief_spot.item_nr)
            for bg in self.belief_grids:
                for v in range(len(bg.data[item_nr])):
                    for u in range(len(bg.data[item_nr, 0])):
                        x, y = bg.get_xy(u, v)
                        bg.data[item_nr, v, u] += prob * (bg.cell_width * bg.cell_height) * (
                                1 / (2 * math.pi * sigma)) * math.exp(
                            -0.5 * ((x - mu_x) ** 2 + (y - mu_y) ** 2) / sigma)
        for item_nr in range(self.nr_of_items):
            # normalize belief
            self.set_b_tot(item_nr, total_belief=1.0)
            self.normalize_belief(item_nr=item_nr)

    # if item_nr = -1 -> belief is normalized for all items
    # b_tot is either a scalar if item_nr != -1 or a list otherwise, containing the normalization constant for the item(s)
    def normalize_belief(self, item_nr=-1, belief_grids=None):
        if belief_grids is None:
            belief_grids = self.belief_grids
        total_belief_per_item = np.array([0.0] * self.nr_of_items)
        for n in range(len(belief_grids)):
            for i in range(self.nr_of_items):
                total_belief_per_item[i] += belief_grids[n].get_aggregated_belief(item_nr=i)
        for bg in belief_grids:
            if item_nr == -1:
                for i in range(self.nr_of_items):
                    bg.normalize_grid(total_belief_pre=total_belief_per_item[i], total_belief_post=self.b_tot[i],
                                      item_nr=i)
            else:
                bg.normalize_grid(total_belief_pre=total_belief_per_item[item_nr],
                                  total_belief_post=self.b_tot[item_nr], item_nr=item_nr)
        self.belief_grids = belief_grids

    def get_belief_vector(self, x, y):
        # get belief grid
        belief_vector = [0] * self.nr_of_items
        bg = self.get_belief_grid(x, y)
        if bg == 'outside':
            return belief_vector
        for i in range(self.nr_of_items):
            belief_vector[i] = bg.get_belief_value(x, y, item_nr=i)

        return belief_vector

    # observations = [(x0,y0), (x1,y1), ..., (xm, ym)]
    def get_total_belief_of_observations(self, observations):
        total_belief_vector = np.zeros(self.nr_of_items)
        obs_set = set()  # key = (node_nr, bg_nr, u, v), value = belief_vector at that (uniquely identified) cell
        for obs in observations:
            bg_nr = self.get_bg_nr(x=obs[0], y=obs[1])
            if bg_nr != -1:
                u, v = self.belief_grids[bg_nr].get_uv(x=obs[0], y=obs[1])
                if (bg_nr, u, u) not in obs_set:
                    obs_set.add((bg_nr, u, v))
                    total_belief_vector += self.belief_grids[bg_nr].get_belief_vector(u, v)
        return total_belief_vector

    def get_bg_nr(self, x, y):
        for bg_nr in range(len(self.belief_grids)):
            if self.belief_grids[bg_nr].rec.is_point_in_rec(x, y):
                return bg_nr
        return -1

    def get_belief_grid(self, x, y):
        for bg in self.belief_grids:
            if bg.rec.is_point_in_rec(x, y):
                return bg
        return 'outside'

    def get_aggregated_belief(self, rec_nr):
        belief_aggr = [0] * self.nr_of_items
        for i in range(self.nr_of_items):
            # print('i={}, bg = {}'.format(i, self.belief_grids[rec_nr].get_aggregated_belief(item_nr=i)))
            belief_aggr[i] += self.belief_grids[rec_nr].get_aggregated_belief(item_nr=i)
        return np.array(belief_aggr)

    def draw(self, artist_b0, item_nr, world_width, world_height):
        nr_of_cells_x, nr_of_cells_y = int(world_width * 10), int(world_height * 10)
        data = self.create_grid_data(nr_of_cells_x, nr_of_cells_y, world_width, world_height, item_nr)
        artist_b0.set_data(data)

    def draw_plot(self, agent_x, agent_y, item_nr, world_width, world_height):
        fig = plt.figure()
        ax = plt.axes()

        ax.set_xlim(-0.5, world_width + 0.5)
        ax.set_ylim(-0.5, world_height + 0.5)
        ax.invert_yaxis()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('b(x{})'.format(item_nr))
        artist_b0 = ax.imshow([[]], interpolation='none', norm=LogNorm(vmin=10 ** (-10), vmax=1.0),
                              extent=[0, world_width, 0, world_height], zorder=0)
        self.draw(artist_b0, item_nr=item_nr, world_width=world_width, world_height=world_height)
        ax.plot([agent_x], [agent_y], 'ko')
        fig.colorbar(artist_b0)
        plt.show()

    def create_grid_data(self, nr_of_cells_x, nr_of_cells_y, world_width, world_height, item_nr=-1):
        # if item_nr is not specified, get belief grid for all items
        if item_nr == -1:
            data = np.zeros((self.nr_of_items, nr_of_cells_y, nr_of_cells_x))
            for i in range(np.shape(data)[0]):
                for v in range(np.shape(data)[1]):
                    for u in range(np.shape(data)[2]):
                        x = world_width * (u + 0.5) / np.shape(data)[2]
                        y = world_height * (v + 0.5) / np.shape(data)[1]
                        data[i, nr_of_cells_y - v - 1, u] = self.get_belief_vector(x, y)[i]
        else:
            data = np.zeros((nr_of_cells_y, nr_of_cells_x))
            for v in range(np.shape(data)[0]):
                for u in range(np.shape(data)[1]):
                    x = world_width * (u + 0.5) / np.shape(data)[1]
                    y = world_height * (v + 0.5) / np.shape(data)[0]
                    data[nr_of_cells_y - v - 1, u] = self.get_belief_vector(x, y)[item_nr]
        return data


class BeliefRepresentation:
    # node_mapping is a dict with key = node_nr and value = list of rectangles indices belonging to that node
    def __init__(self, belief, nodes_recs_mapping):
        self.node_mapping = nodes_recs_mapping
        self.belief = belief

    def set_belief(self, belief):
        self.belief = belief

    def is_point_in_node(self, x, y, node_nr):
        for idx in self.node_mapping[node_nr]:
            if self.belief.belief_grids[idx].rec.is_point_in_rec(x, y):
                return True
        return False

    def get_node_nr(self, x, y):
        for n in range(len(self.node_mapping.keys())):
            rec_indices = self.node_mapping[n]
            for idx in rec_indices:
                if self.belief.belief_grids[idx].rec.is_point_in_rec(x, y):
                    return n
        return -1

    def get_aggregated_belief(self, node_nr):
        belief_aggr = [0] * self.belief.nr_of_items
        rec_indices = self.node_mapping[node_nr]
        for idx in rec_indices:
            for i in range(self.belief.nr_of_items):
                belief_aggr[i] += self.belief.belief_grids[idx].get_aggregated_belief(item_nr=i)
        return np.array(belief_aggr)

    def get_median_belief(self, node_nr):
        rec_indices = self.node_mapping[node_nr]
        all_data = []
        for idx in rec_indices:
            all_data += list(self.belief.belief_grids[idx].data.flatten())
        return np.median(all_data)

    def get_N_max_belief_values(self, node_nr, N):
        max_values = []
        rec_indices = self.node_mapping[node_nr]
        for idx in rec_indices:
            if self.belief.belief_grids[idx].data.shape[1] * self.belief.belief_grids[idx].data.shape[2] < 100:
                N = int(self.belief.belief_grids[idx].data.shape[1] * self.belief.belief_grids[idx].data.shape[2] / 2)
            max_values += list(np.partition(self.belief.belief_grids[idx].data.flatten(), -N))
        max_values = np.array(max_values)
        max_values = max_values[np.argsort(max_values)[-N:]]
        return max_values

    def get_max_belief_cell_value(self, node_nr, item_nr=-1):
        b_max = 0
        rec_indices = self.node_mapping[node_nr]
        for idx in rec_indices:
            if item_nr == -1:
                bg_max = np.max(self.belief.belief_grids[idx].data)
            else:
                bg_max = np.max(self.belief.belief_grids[idx].data[item_nr])
            if bg_max > b_max:
                b_max = bg_max
        return b_max

    def get_max_belief_cell_pos(self, node_nr, item_nr):
        b_max = 0
        rec_indices = self.node_mapping[node_nr]
        for idx in rec_indices:
            bg = self.belief.belief_grids[idx]
            bg_max = np.max(bg.data)
            if bg_max > b_max:
                b_max = bg_max
                bmax_v, bmax_u = np.unravel_index(bg.data[item_nr].argmax(), bg.data[item_nr].shape)
                bmax_x, bmax_y = bg.get_xy(u=bmax_u, v=bmax_v)
        return bmax_x, bmax_y
