import math
import numpy as np


class Rectangle:
    def __init__(self, x0=-1, y0=-1, width=-1, height=-1):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

    def set_rectangle(self, x0, y0, width, height):
        self.x0 = x0
        self.y0 = y0
        self.width = width
        self.height = height

    def is_point_in_rec(self, p_x, p_y):
        return (self.x0 <= p_x < self.x0 + self.width) and (self.y0 <= p_y < self.y0 + self.height)


    # x_array and y_array are numpy arrays
    def are_points_in_rec(self, x_array, y_array):
        x_cond = np.logical_and(self.x0 <= x_array, x_array < self.x0 + self.width)
        y_cond = np.logical_and(self.y0 <= y_array, y_array < self.y0 + self.height)
        return np.logical_and(x_cond, y_cond)

    # x and y are scalars
    def get_dist_to_rec(self, x, y):
        dist = -1
        dx = min(math.fabs(x - self.x0), math.fabs(x - self.x0 - self.width))
        dy = min(math.fabs(y - self.y0), math.fabs(y - self.y0 - self.height))
        if (x < self.x0 or x > self.x0 + self.width) and (y < self.y0 or y > self.y0 + self.height):
            dist = math.sqrt(dx**2 + dy**2)
        elif (self.x0 <= x < self.x0 + self.width) and (y < self.y0 or y > self.y0 + self.height):
            dist = dy
        elif (self.y0 <= y < self.y0 + self.height) and (x < self.x0 or x > self.x0 + self.width):
            dist = dx
        else:
            dist = min(dx, dy)
        return dist

    # x and y are numpy array, returns numpy array dist
    def get_dist_to_rec_numpy(self, x, y):
        dist = np.zeros(shape=np.shape(x))
        dx = np.array([np.fabs(x-self.x0), np.fabs(x - self.x0 - self.width)]).min(axis=0)
        dy = np.array([np.fabs(y-self.y0), np.fabs(y - self.y0 - self.height)]).min(axis=0)
        within_width = np.logical_and(self.x0 <= x, x < self.x0 + self.width)
        within_height = np.logical_and(self.y0 <= y, y < self.y0 + self.height)
        not_within_both = np.logical_and(np.logical_not(within_width), np.logical_not(within_height))
        within_width_not_height = np.logical_and(within_width, np.logical_not(within_height))
        within_height_not_width = np.logical_and(np.logical_not(within_width), within_height)
        within_both = np.logical_and(within_width, within_height)

        dist[not_within_both] = np.sqrt(dx[not_within_both]**2 + dy[not_within_both]**2)
        dist[within_width_not_height] = dy[within_width_not_height]
        dist[within_height_not_width] = dx[within_height_not_width]
        dist[within_both] = np.array([dx[within_both], dy[within_both]]).min(axis=0)

        return dist

    def get_mid_point(self):
        return self.x0 + self.width/2., self.y0 + self.height/2.

    def get_area(self):
        return self.width * self.height

    def get_intersection_rec(self, rec):
        # first check if the recs intersect at all
        do_recs_intersect = ((self.x0 <= rec.x0 < self.x0 + self.width)) and (
        (self.y0 <= rec.y0 < self.y0 + self.height)) or \
                            ((rec.x0 <= self.x0 < rec.x0 + rec.width) and (rec.y0 <= self.y0 < rec.y0 + rec.height))
        if not do_recs_intersect:
            return 'no'
        x0_in = max(self.x0, rec.x0)
        y0_in = max(self.y0, rec.y0)
        width_in = min(self.x0+self.width, rec.x0+rec.width) - x0_in
        height_in = min(self.y0+self.height, rec.y0+rec.height) - y0_in
        return Rectangle(x0=x0_in, y0=y0_in, width=width_in, height=height_in)
