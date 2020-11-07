import math

import matplotlib.pyplot as plt


def inverse_item_map(item_nr, item_map):
    for item_type in item_map.keys():
        if item_map[item_type] == item_nr:
            return item_type
    return 'none'


def get_discount_factor():
    return 0.999


# return input angle s.t. it is in the interval (-pi, pi]
def angle_consistency(angle):
    angle %= (2 * math.pi)
    if angle > math.pi:
        angle -= 2 * math.pi
    if angle <= - math.pi:
        angle += 2 * math.pi
    return angle


def deg_to_rad(angle):
    return angle * math.pi / 180


def rad_to_deg(angle):
    return angle / math.pi * 180.0


def nav_action_get_n0(a_name):
    after_bracket = a_name.partition('(')[2]
    n0_str = after_bracket.partition(',')[0]
    return int(n0_str)


def nav_action_get_n1(a_name):
    after_comma = a_name.partition(',')[2]
    n1_str = after_comma.partition(')')[0]
    return int(n1_str)


def pickup_get_i(a_name):
    return int(a_name[6:])


def get_lines(rec, lines):
    line_x1 = [rec.x0, rec.x0 + rec.width]
    line_x2 = [rec.x0 + rec.width, rec.x0 + rec.width]
    line_x3 = [rec.x0 + rec.width, rec.x0]
    line_x4 = [rec.x0, rec.x0]

    line_y1 = [rec.y0, rec.y0]
    line_y2 = [rec.y0, rec.y0 + rec.height]
    line_y3 = [rec.y0 + rec.height, rec.y0 + rec.height]
    line_y4 = [rec.y0 + rec.height, rec.y0]

    lines.append([line_x1, line_y1])
    lines.append([line_x2, line_y2])
    lines.append([line_x3, line_y3])
    lines.append([line_x4, line_y4])


# very ugly code, there must be a much more elegant solution. but it works for now
def get_nodes_recs_mapping(layer, nr_of_layers, node_mapping, nodegraph_lN):
    node_rec_mapping = {}
    if layer == nr_of_layers - 1:
        counter = 0
        for node, recs in nodegraph_lN.node_recs.items():
            node_rec_mapping[node] = [counter + i for i in range(len(recs))]
            counter = node_rec_mapping[node][-1] + 1
    else:
        for node in node_mapping[layer]:
            node_rec_mapping[node] = node_mapping[layer][node]
        for idx in range(layer + 1, nr_of_layers):
            if idx == nr_of_layers - 1:
                recs_list = nodegraph_lN.get_all_recs_as_flat_list()
                for top_node in node_rec_mapping:
                    rec_indices = []
                    for node in node_rec_mapping[top_node]:
                        for rec in nodegraph_lN.node_recs[node]:
                            rec_indices += [recs_list.index(rec)]
                    node_rec_mapping[top_node] = rec_indices
            else:
                for top_node in node_rec_mapping:
                    nodes_new = []
                    for node_new in node_rec_mapping[top_node]:
                        nodes_new += node_mapping[idx][node_new]
                    node_rec_mapping[top_node] = nodes_new
    return node_rec_mapping


def draw_rectangles(recs):
    plt.figure()
    ax = plt.axes()
    ax.invert_yaxis()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', 'box')
    lines = []
    for rec in recs:
        get_lines(rec, lines)
    for line in lines:
        ax.plot(line[0], line[1], color='black', linewidth=1)
    plt.show()
