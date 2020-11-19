import math
import logging

class Item:
    def __init__(self, item_type='none', x=-1, y=-1, u=-1, v=-1):
        self.log = logging.getLogger(__name__)
        self.item_type = item_type
        self.x = x
        self.y = y
        self.u = u
        self.v = v

    def get_distance(self, x, y, cell_width=-1, cell_height=-1):
        if self.x == -1 or self.y == -1:
            if self.u == -1 or self.v == -1 or cell_height == -1 or cell_width == -1:
                print('ERROR: item is not initialized properly, need to know either (x,y) or (u,v)')
                self.log.warning('ERROR: item is not initialized properly, need to know either (x,y) or (u,v)')
            else:
                self.x = (self.u + 0.5) * cell_width
                self.y = (self.v + 0.5) * cell_height

        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2)

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_index(self, u, v, cell_width, cell_height):
        self.u = u
        self.v = v
        self.x = (u + 0.5) * cell_width
        self.y = (v + 0.5) * cell_height

    def set_item_type(self, item_type):
        self.item_type = item_type
