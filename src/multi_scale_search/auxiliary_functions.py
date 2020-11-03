import math
import matplotlib.pyplot as plt
from src.auxiliary_files.rectangle import Rectangle
import config


def inverse_item_map(item_nr, item_map):
    for item_type in item_map.keys():
        if item_map[item_type] == item_nr:
            return item_type
    return 'none'


def get_discount_factor(environment_type):
    return 0.999


def get_timeout_time(environment_type, solver_type):
    if config.simulation_name == 'low_fidelity_simulation':
        if environment_type == 'small':
            if solver_type in config.multiscale_agent_types:
                return 5.0
            else:
                return 5.0
        elif environment_type == 'big':
            if solver_type in config.multiscale_agent_types:
                return 10.0
            else:
                return 10.0


def get_pickup_reward(environment_type):
    if environment_type == 'small':
        return 20
    elif environment_type == 'big':
        return 50


def get_delivery_reward(environment_type):
    if environment_type == 'small':
        return 200
    elif environment_type == 'big':
        return 250


def get_penalty_reward(environment_type):
    if environment_type == 'small':
        return -5.0
    elif environment_type == 'big':
        return -0.0


def get_agent_types(environment_type='small'):
    if environment_type == 'small':
        return config.agent_types
    elif environment_type == 'big':
        return config.agent_types


def get_world_dimensions(environment_type='small'):
    if environment_type == 'small':
        return 14.0, 13.4
    elif environment_type == 'big':
        return 28.0, 26.0
    else:
        print('unknown environment_type={} in config.get_world_dimensions()'.format(environment_type))


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


def get_recs_of_nodes(rec_env, environment='small', nr_of_layers=2):
    if environment == 'small':
        if nr_of_layers == 2 or nr_of_layers == 1:
            recs_n0 = [Rectangle(rec_env[0].x0, rec_env[0].y0, rec_env[0].width / 2, rec_env[0].height)]
            recs_n1 = [Rectangle(rec_env[0].x0 + rec_env[0].width / 2, rec_env[0].y0, rec_env[0].width / 2,
                                 rec_env[0].height)]
            recs_n2 = [Rectangle(rec_env[1].x0, rec_env[1].y0, rec_env[1].width, rec_env[1].height)]
            recs_n3 = [Rectangle(rec_env[2].x0, rec_env[2].y0, rec_env[2].width, rec_env[2].height / 2.0)]
            recs_n4 = [Rectangle(rec_env[2].x0, rec_env[2].y0 + rec_env[2].height / 2.0, rec_env[2].width,
                                 rec_env[2].height / 2.0)]
            recs_n5 = [Rectangle(rec_env[3].x0, rec_env[3].y0, rec_env[3].width, rec_env[3].height)]
            recs_n6 = [Rectangle(rec_env[4].x0, rec_env[4].y0 + 1.0 * rec_env[4].height / 3.0, rec_env[4].width,
                                 rec_env[4].height / 3.0)]
            recs_n7 = [Rectangle(rec_env[4].x0, rec_env[4].y0, rec_env[4].width, rec_env[4].height / 3.0)]
            recs_n8 = [Rectangle(rec_env[4].x0, rec_env[4].y0 + 2.0 * rec_env[4].height / 3.0, rec_env[4].width,
                                 rec_env[4].height / 3.0)]
            return {0: recs_n0, 1: recs_n1, 2: recs_n2, 3: recs_n3, 4: recs_n4, 5: recs_n5, 6: recs_n6, 7: recs_n7,
                    8: recs_n8}
    elif environment == 'big':
        if nr_of_layers < 3:
            print('Error: need at least 3 layers for the big environment')
        if nr_of_layers == 3 or nr_of_layers == 4:
            # hallways + big room upper half ###########################################################################
            recs_n0 = [Rectangle(x0=rec_env[8].x0, y0=rec_env[8].y0, width=2., height=4.)]
            recs_n1 = [Rectangle(x0=rec_env[4].x0, y0=rec_env[4].y0, width=rec_env[4].width, height=rec_env[4].height),
                       Rectangle(x0=recs_n0[0].x0 + recs_n0[0].width, y0=recs_n0[0].y0, width=2., height=4.)]
            recs_n2 = [Rectangle(x0=recs_n1[1].x0 + recs_n1[1].width, y0=recs_n0[0].y0, width=2.5, height=4.0)]
            recs_n3 = [Rectangle(x0=recs_n2[0].x0 + recs_n2[0].width, y0=recs_n0[0].y0, width=2.5, height=4.0)]
            recs_n4 = [Rectangle(x0=rec_env[5].x0, y0=rec_env[5].y0, width=rec_env[5].width, height=rec_env[5].height),
                       Rectangle(x0=recs_n3[0].x0 + recs_n3[0].width, y0=recs_n0[0].y0, width=2., height=2.0)]
            recs_n5 = [Rectangle(x0=recs_n4[1].x0 + recs_n4[1].width, y0=recs_n0[0].y0, width=2.5, height=2.0)]
            recs_n6 = [Rectangle(x0=recs_n5[0].x0 + recs_n5[0].width, y0=recs_n0[0].y0, width=2.5, height=2.0)]
            recs_n7 = [Rectangle(x0=rec_env[6].x0, y0=rec_env[6].y0, width=rec_env[6].width, height=rec_env[6].height),
                       Rectangle(x0=recs_n6[0].x0 + recs_n6[0].width, y0=recs_n0[0].y0, width=2., height=2.0)]
            recs_n8 = [Rectangle(x0=recs_n7[1].x0 + recs_n7[1].width, y0=recs_n0[0].y0, width=2.5, height=4.0)]
            recs_n9 = [Rectangle(x0=recs_n8[0].x0 + recs_n8[0].width, y0=recs_n0[0].y0, width=2.5, height=4.0)]
            recs_n10 = [Rectangle(x0=rec_env[7].x0, y0=rec_env[7].y0, width=rec_env[7].width, height=rec_env[7].height),
                        Rectangle(x0=recs_n9[0].x0 + recs_n9[0].width, y0=recs_n0[0].y0, width=2., height=4.0)]
            recs_n11 = [Rectangle(x0=recs_n10[1].x0 + recs_n10[1].width, y0=recs_n0[0].y0, width=2., height=4.0)]
            # hallways + big room bottom half ##########################################################################
            recs_n12 = [Rectangle(x0=rec_env[8].x0, y0=rec_env[8].y0 + recs_n0[0].height, width=2., height=4.)]
            recs_n13 = [Rectangle(x0=rec_env[9].x0, y0=rec_env[9].y0, width=rec_env[9].width, height=rec_env[9].height),
                        Rectangle(x0=recs_n12[0].x0 + recs_n12[0].width, y0=recs_n12[0].y0, width=2., height=4.)]
            recs_n14 = [Rectangle(x0=recs_n13[1].x0 + recs_n13[1].width, y0=recs_n12[0].y0, width=2.5, height=4.0)]
            recs_n15 = [Rectangle(x0=recs_n14[0].x0 + recs_n14[0].width, y0=recs_n12[0].y0, width=2.5, height=4.0)]
            recs_n16 = [
                Rectangle(x0=rec_env[10].x0, y0=rec_env[10].y0, width=rec_env[10].width, height=rec_env[10].height),
                Rectangle(x0=recs_n15[0].x0 + recs_n15[0].width, y0=recs_n12[0].y0 + 2., width=2., height=2.0)]
            recs_n17 = [Rectangle(x0=recs_n16[1].x0 + recs_n16[1].width, y0=recs_n12[0].y0 + 2., width=2.5, height=2.0)]
            recs_n18 = [Rectangle(x0=recs_n17[0].x0 + recs_n17[0].width, y0=recs_n12[0].y0 + 2., width=2.5, height=2.0)]
            recs_n19 = [
                Rectangle(x0=rec_env[11].x0, y0=rec_env[11].y0, width=rec_env[11].width, height=rec_env[11].height),
                Rectangle(x0=recs_n18[0].x0 + recs_n18[0].width, y0=recs_n12[0].y0 + 2., width=2., height=2.0)]
            recs_n20 = [Rectangle(x0=recs_n19[1].x0 + recs_n19[1].width, y0=recs_n12[0].y0, width=2.5, height=4.0)]
            recs_n21 = [Rectangle(x0=recs_n20[0].x0 + recs_n20[0].width, y0=recs_n12[0].y0, width=2.5, height=4.0)]
            recs_n22 = [
                Rectangle(x0=rec_env[12].x0, y0=rec_env[12].y0, width=rec_env[12].width, height=rec_env[12].height),
                Rectangle(x0=recs_n21[0].x0 + recs_n21[0].width, y0=recs_n12[0].y0, width=2., height=4.0)]
            recs_n23 = [Rectangle(x0=recs_n22[1].x0 + recs_n22[1].width, y0=recs_n12[0].y0, width=2., height=4.0)]
            # TOP 4 Rooms ##############################################################################################
            # room 1
            recs_n24 = [Rectangle(x0=rec_env[0].x0, y0=rec_env[0].y0, width=2., height=6.)]
            recs_n25 = [Rectangle(x0=recs_n24[0].x0 + recs_n24[0].width, y0=recs_n24[0].y0, width=2., height=6.0)]
            recs_n26 = [Rectangle(x0=recs_n25[0].x0 + recs_n25[0].width, y0=recs_n25[0].y0, width=2., height=6.0)]
            recs_n27 = [Rectangle(x0=recs_n24[0].x0, y0=recs_n24[0].y0 + recs_n24[0].height, width=6., height=2.)]
            # room 2
            recs_n28 = [Rectangle(x0=rec_env[1].x0, y0=rec_env[1].y0, width=2., height=6.)]
            recs_n29 = [Rectangle(x0=recs_n28[0].x0 + recs_n28[0].width, y0=recs_n28[0].y0, width=2., height=6.0)]
            recs_n30 = [Rectangle(x0=recs_n29[0].x0 + recs_n29[0].width, y0=recs_n29[0].y0, width=2., height=6.0)]
            recs_n31 = [Rectangle(x0=recs_n28[0].x0, y0=recs_n28[0].y0 + recs_n28[0].height, width=6., height=2.)]
            # room 3
            recs_n32 = [Rectangle(x0=rec_env[2].x0, y0=rec_env[2].y0, width=2., height=6.)]
            recs_n33 = [Rectangle(x0=recs_n32[0].x0 + recs_n32[0].width, y0=recs_n32[0].y0, width=2., height=6.0)]
            recs_n34 = [Rectangle(x0=recs_n33[0].x0 + recs_n33[0].width, y0=recs_n33[0].y0, width=2., height=6.0)]
            recs_n35 = [Rectangle(x0=recs_n32[0].x0, y0=recs_n32[0].y0 + recs_n32[0].height, width=6., height=2.)]
            # room 4
            recs_n36 = [Rectangle(x0=rec_env[3].x0, y0=rec_env[3].y0, width=2., height=6.)]
            recs_n37 = [Rectangle(x0=recs_n36[0].x0 + recs_n36[0].width, y0=recs_n36[0].y0, width=2., height=6.0)]
            recs_n38 = [Rectangle(x0=recs_n37[0].x0 + recs_n37[0].width, y0=recs_n37[0].y0, width=2., height=6.0)]
            recs_n39 = [Rectangle(x0=recs_n36[0].x0, y0=recs_n36[0].y0 + recs_n36[0].height, width=6., height=2.)]
            # BOTTOM 4 rooms ###########################################################################################
            # room 5
            recs_n40 = [Rectangle(x0=rec_env[13].x0, y0=rec_env[13].y0, width=6., height=2.)]
            recs_n41 = [Rectangle(x0=recs_n40[0].x0, y0=recs_n40[0].y0 + recs_n40[0].height, width=2., height=6.)]
            recs_n42 = [Rectangle(x0=recs_n41[0].x0 + recs_n41[0].width, y0=recs_n41[0].y0, width=2., height=6.)]
            recs_n43 = [Rectangle(x0=recs_n42[0].x0 + recs_n42[0].width, y0=recs_n41[0].y0, width=2., height=6.)]
            # room 6
            recs_n44 = [Rectangle(x0=rec_env[14].x0, y0=rec_env[14].y0, width=6., height=2.)]
            recs_n45 = [Rectangle(x0=recs_n44[0].x0, y0=recs_n44[0].y0 + recs_n44[0].height, width=2., height=6.)]
            recs_n46 = [Rectangle(x0=recs_n45[0].x0 + recs_n45[0].width, y0=recs_n45[0].y0, width=2., height=6.)]
            recs_n47 = [Rectangle(x0=recs_n46[0].x0 + recs_n46[0].width, y0=recs_n45[0].y0, width=2., height=6.)]
            # room 7
            recs_n48 = [Rectangle(x0=rec_env[15].x0, y0=rec_env[15].y0, width=6., height=2.)]
            recs_n49 = [Rectangle(x0=recs_n48[0].x0, y0=recs_n48[0].y0 + recs_n48[0].height, width=2., height=6.)]
            recs_n50 = [Rectangle(x0=recs_n49[0].x0 + recs_n49[0].width, y0=recs_n49[0].y0, width=2., height=6.)]
            recs_n51 = [Rectangle(x0=recs_n50[0].x0 + recs_n50[0].width, y0=recs_n49[0].y0, width=2., height=6.)]
            # room 8
            recs_n52 = [Rectangle(x0=rec_env[16].x0, y0=rec_env[16].y0, width=6., height=2.)]
            recs_n53 = [Rectangle(x0=recs_n52[0].x0, y0=recs_n52[0].y0 + recs_n52[0].height, width=2., height=6.)]
            recs_n54 = [Rectangle(x0=recs_n53[0].x0 + recs_n53[0].width, y0=recs_n53[0].y0, width=2., height=6.)]
            recs_n55 = [Rectangle(x0=recs_n54[0].x0 + recs_n54[0].width, y0=recs_n53[0].y0, width=2., height=6.)]
            if nr_of_layers == 3:
                return {0: recs_n0, 1: recs_n1, 2: recs_n2, 3: recs_n3, 4: recs_n4, 5: recs_n5, 6: recs_n6, 7: recs_n7,
                        8: recs_n8, 9: recs_n9, 10: recs_n10, 11: recs_n11, 12: recs_n12, 13: recs_n13, 14: recs_n14,
                        15: recs_n15, 16: recs_n16, 17: recs_n17, 18: recs_n18, 19: recs_n19, 20: recs_n20,
                        21: recs_n21,
                        22: recs_n22, 23: recs_n23, 24: recs_n24, 25: recs_n25, 26: recs_n26, 27: recs_n27,
                        28: recs_n28,
                        29: recs_n29, 30: recs_n30, 31: recs_n31, 32: recs_n32, 33: recs_n33, 34: recs_n34,
                        35: recs_n35,
                        36: recs_n36, 37: recs_n37, 38: recs_n38, 39: recs_n39, 40: recs_n40, 41: recs_n41,
                        42: recs_n42,
                        43: recs_n43, 44: recs_n44, 45: recs_n45, 46: recs_n46, 47: recs_n47, 48: recs_n48,
                        49: recs_n49,
                        50: recs_n50, 51: recs_n51, 52: recs_n52, 53: recs_n53, 54: recs_n54, 55: recs_n55}
        if nr_of_layers == 4:
            print('4 layers not implemented yet')
    else:
        print('unknown environment type')


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


def construct_nodegraph_multiscale(nodegraph, layer, environment='small'):
    if environment == 'small':
        if layer == 0:
            nodegraph.add_edge(0, 1)
            nodegraph.add_edge(1, 2)
        elif layer == 1:
            nodegraph.add_edge(0, 1)
            nodegraph.add_edge(0, 2)
            nodegraph.add_edge(1, 2)
            nodegraph.add_edge(2, 3)
            nodegraph.add_edge(3, 4)
            nodegraph.add_edge(3, 5)
            nodegraph.add_edge(4, 5)
            nodegraph.add_edge(5, 6)
            nodegraph.add_edge(6, 7)
            nodegraph.add_edge(6, 8)
    elif environment == 'big':
        if layer == 0:
            nodegraph.add_edge(0, 1)
            nodegraph.add_edge(0, 2)
            nodegraph.add_edge(1, 3)
            nodegraph.add_edge(2, 3)
        elif layer == 1:
            nodegraph.add_edge(0, 1)
            nodegraph.add_edge(1, 2)
            nodegraph.add_edge(1, 9)
            nodegraph.add_edge(1, 10)
            nodegraph.add_edge(2, 3)
            nodegraph.add_edge(2, 5)
            nodegraph.add_edge(2, 9)
            nodegraph.add_edge(2, 10)
            nodegraph.add_edge(4, 5)
            nodegraph.add_edge(5, 6)
            nodegraph.add_edge(5, 13)
            nodegraph.add_edge(5, 14)
            nodegraph.add_edge(6, 7)
            nodegraph.add_edge(6, 13)
            nodegraph.add_edge(6, 14)
            nodegraph.add_edge(8, 9)
            nodegraph.add_edge(9, 10)
            nodegraph.add_edge(10, 11)
            nodegraph.add_edge(10, 13)
            nodegraph.add_edge(12, 13)
            nodegraph.add_edge(13, 14)
            nodegraph.add_edge(14, 15)
        elif layer == 2:
            nodegraph.add_edge(0, 1)
            nodegraph.add_edge(0, 12)
            nodegraph.add_edge(0, 13)
            nodegraph.add_edge(1, 2)
            nodegraph.add_edge(1, 12)
            nodegraph.add_edge(1, 13)
            nodegraph.add_edge(1, 14)
            nodegraph.add_edge(1, 27)
            nodegraph.add_edge(2, 3)
            nodegraph.add_edge(2, 13)
            nodegraph.add_edge(2, 14)
            nodegraph.add_edge(2, 15)
            nodegraph.add_edge(3, 4)
            nodegraph.add_edge(3, 14)
            nodegraph.add_edge(3, 15)
            nodegraph.add_edge(4, 5)
            nodegraph.add_edge(4, 31)
            nodegraph.add_edge(5, 6)
            nodegraph.add_edge(6, 7)
            nodegraph.add_edge(7, 8)
            nodegraph.add_edge(7, 35)
            nodegraph.add_edge(8, 9)
            nodegraph.add_edge(8, 20)
            nodegraph.add_edge(8, 21)
            nodegraph.add_edge(9, 10)
            nodegraph.add_edge(9, 20)
            nodegraph.add_edge(9, 21)
            nodegraph.add_edge(9, 22)
            nodegraph.add_edge(10, 11)
            nodegraph.add_edge(10, 21)
            nodegraph.add_edge(10, 22)
            nodegraph.add_edge(10, 23)
            nodegraph.add_edge(10, 39)
            nodegraph.add_edge(11, 22)
            nodegraph.add_edge(11, 23)
            nodegraph.add_edge(12, 13)
            nodegraph.add_edge(13, 14)
            nodegraph.add_edge(13, 40)
            nodegraph.add_edge(14, 15)
            nodegraph.add_edge(15, 16)
            nodegraph.add_edge(16, 17)
            nodegraph.add_edge(16, 44)
            nodegraph.add_edge(17, 18)
            nodegraph.add_edge(18, 19)
            nodegraph.add_edge(19, 20)
            nodegraph.add_edge(19, 48)
            nodegraph.add_edge(20, 21)
            nodegraph.add_edge(21, 22)
            nodegraph.add_edge(22, 23)
            nodegraph.add_edge(22, 52)
            nodegraph.add_edge(24, 25)
            nodegraph.add_edge(24, 27)
            nodegraph.add_edge(25, 26)
            nodegraph.add_edge(25, 27)
            nodegraph.add_edge(26, 27)
            nodegraph.add_edge(28, 29)
            nodegraph.add_edge(28, 31)
            nodegraph.add_edge(29, 30)
            nodegraph.add_edge(29, 31)
            nodegraph.add_edge(30, 31)
            nodegraph.add_edge(32, 33)
            nodegraph.add_edge(32, 35)
            nodegraph.add_edge(33, 34)
            nodegraph.add_edge(33, 35)
            nodegraph.add_edge(34, 35)
            nodegraph.add_edge(36, 37)
            nodegraph.add_edge(36, 39)
            nodegraph.add_edge(37, 38)
            nodegraph.add_edge(37, 39)
            nodegraph.add_edge(38, 39)
            nodegraph.add_edge(40, 41)
            nodegraph.add_edge(40, 42)
            nodegraph.add_edge(40, 43)
            nodegraph.add_edge(41, 42)
            nodegraph.add_edge(42, 43)
            nodegraph.add_edge(44, 45)
            nodegraph.add_edge(44, 46)
            nodegraph.add_edge(44, 47)
            nodegraph.add_edge(45, 46)
            nodegraph.add_edge(46, 47)
            nodegraph.add_edge(48, 49)
            nodegraph.add_edge(48, 50)
            nodegraph.add_edge(48, 51)
            nodegraph.add_edge(49, 50)
            nodegraph.add_edge(50, 51)
            nodegraph.add_edge(52, 53)
            nodegraph.add_edge(52, 54)
            nodegraph.add_edge(52, 55)
            nodegraph.add_edge(53, 54)
            nodegraph.add_edge(54, 55)
    else:
        print('env={} is not implemented in function construct_nodegraph_MultiScale'.format(environment))


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
