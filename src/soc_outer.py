from fractions import Fraction
import math

def define_soc_constraints(network, cone_budget):
    if cone_budget == 1:
        single_hydraulic_loss_soc(network.pipes | network.valves, network.time_steps)
        single_head_gain_soc(network.pumps, network.time_steps)
    else:
        multiple_hydraulic_loss_soc(network.pipes | network.valves, network.time_steps, cone_budget)
        multiple_head_gain_soc(network.pumps, network.time_steps, cone_budget)

def single_hydraulic_loss_soc(links, time_steps):
    for link in links.values():
        link.soc_outer_approximation = {}
        for t in range(1, time_steps + 1):
            if link.fixed_flow[t-1] or (hasattr(link, 'status_decisions') and link.status_decisions[t-1] == 0) or (hasattr(link, 'selection_decision') and link.selection_decision == 0): continue
            if link.flow_upper_bound[t - 1] > 0.0 and (not hasattr(link, "direction_decision") or link.direction_decision[t-1] != 0):
                link.soc_outer_approximation[t,'pos'] = {'method': 'single', 'exponent': 1.0 / link.exponent, 'x_coef': 1.0, 'y_coef': 1.0, 'z_coef': link.resistance_coefficient ** (1 / link.exponent), 'x_bounds': [min(0.0, link.start_node.head_lower_bound[t-1] - link.end_node.head_upper_bound[t-1]), max(0.0, link.start_node.head_upper_bound[t-1] - link.end_node.head_lower_bound[t-1])], 'y_bounds': [0.0, 1.0], 'z_bounds': [link.positive_flow_lower_bound[t-1], link.flow_upper_bound[t-1]]}
            if (hasattr(link, "direction_decision") and link.direction_decision[t-1] != 1 and link.flow_lower_bound[t-1] < 0.0):
                link.soc_outer_approximation[t,'neg'] = {'method': 'single', 'exponent': 1.0 / link.exponent, 'x_coef': 1.0, 'y_coef': 1.0, 'z_coef': link.resistance_coefficient ** (1 / link.exponent), 'x_bounds': [min(0.0, link.end_node.head_lower_bound[t-1] - link.start_node.head_upper_bound[t-1]), max(0.0, link.end_node.head_upper_bound[t-1] - link.start_node.head_lower_bound[t-1])], 'y_bounds': [0.0, 1.0], 'z_bounds': [link.flow_lower_bound[t-1], link.negative_flow_upper_bound[t-1]]}

def single_head_gain_soc(pumps, time_steps):
    for pump in pumps.values():
        pump.soc_outer_approximation = {}
        for t in range(1, time_steps+1):
            if pump.status_decisions[t-1] == 0 or pump.fixed_flow[t-1] or pump.curve_c == 1: continue
            pump.soc_outer_approximation[t] = {'method': 'single', 'exponent': 1.0 / pump.curve_c, 'x_coef': 1.0, 'y_coef': 1.0, 'z_coef': (-pump.curve_b) ** (1 / pump.curve_c), 'x_bounds': [min(0.0, pump.start_node.head_lower_bound[t-1] - pump.end_node.head_upper_bound[t-1] + pump.curve_a), max(0.0, pump.start_node.head_upper_bound[t-1] - pump.end_node.head_lower_bound[t-1] + pump.curve_a)], 'y_bounds': [0.0, 1.0], 'z_bounds': [pump.positive_flow_lower_bound[t-1], pump.flow_upper_bound[t-1]]}

def obtain_soc_lhs(cone_budget, exponent):
    exponent = Fraction(str(exponent))
    largest_rational = Fraction(0, 1)

    for q in range(2, 2**cone_budget + 1):
        p = math.floor(q * exponent)
        if Fraction(p, q) > largest_rational:
            largest_rational = Fraction(p, q)

    if largest_rational == 0:
        return largest_rational, []

    p, q = largest_rational.numerator, largest_rational.denominator
    cone_count = (q - 1).bit_length()

    lhs_cons = []
    pairs = {'x': p, 'y': q - p, 'aux_0': (2**cone_count) - q}

    for k in range(cone_count, 0, -1):
        i, j = [key for key, value in pairs.items() if value % 2 == 1]
        l = next(x for x in pairs.keys() if x not in (i, j))

        if pairs[i] > pairs[j]:
            i, j = j, i

        lhs_cons.append([i, j, f'aux_{k-1}'])
        pairs[j] = (pairs[j] - pairs[i]) // 2
        pairs[l] = pairs[l] // 2
        pairs[f'aux_{k-1}'] = pairs.pop(i)

    return largest_rational, lhs_cons

def multiple_hydraulic_loss_soc(links, time_steps, cone_budget):
    exponent = 1 / Fraction(str(next(iter(links.values())).exponent))
    largest_rational, lhs_cons = obtain_soc_lhs(cone_budget, exponent)

    if largest_rational == 0:
        single_hydraulic_loss_soc(links, time_steps)
        return

    for link in links.values():
        link.soc_outer_approximation = {}

        for t in range(1, time_steps + 1):
            if link.fixed_flow[t-1] or (hasattr(link, 'status_decisions') and link.status_decisions[t-1] == 0) or (hasattr(link, 'selection_decision') and link.selection_decision == 0): continue
            if link.flow_upper_bound[t - 1] > 0.0 and (not hasattr(link, "direction_decision") or link.direction_decision[t-1] != 0):
                z_coef = (link.resistance_coefficient ** largest_rational) * (link.flow_upper_bound[t-1] ** ((largest_rational / exponent) - 1))
                link.soc_outer_approximation[t,'pos'] = {'method': 'multiple', 'lhs_cons': lhs_cons, 'exponent': largest_rational, 'x_coef': 1.0, 'y_coef': 1.0, 'z_coef': z_coef, 'x_bounds': [min(0.0, link.start_node.head_lower_bound[t-1] - link.end_node.head_upper_bound[t-1]), max(0.0, link.start_node.head_upper_bound[t-1] - link.end_node.head_lower_bound[t-1])], 'y_bounds': [0.0, 1.0], 'z_bounds': [link.positive_flow_lower_bound[t-1], link.flow_upper_bound[t-1]]}
            if (hasattr(link, "direction_decision") and link.direction_decision[t-1] != 1 and link.flow_lower_bound[t-1] < 0.0):
                z_coef = (link.resistance_coefficient ** largest_rational) * ((-link.flow_lower_bound[t-1]) ** ((largest_rational / exponent) - 1))
                link.soc_outer_approximation[t,'neg'] = {'method': 'multiple', 'lhs_cons': lhs_cons, 'exponent': largest_rational, 'x_coef': 1.0, 'y_coef': 1.0, 'z_coef': z_coef, 'x_bounds': [min(0.0, link.end_node.head_lower_bound[t-1] - link.start_node.head_upper_bound[t-1]), max(0.0, link.end_node.head_upper_bound[t-1] - link.start_node.head_lower_bound[t-1])], 'y_bounds': [0.0, 1.0], 'z_bounds': [link.flow_lower_bound[t-1], link.negative_flow_upper_bound[t-1]]}

def multiple_head_gain_soc(pumps, time_steps, cone_budget):
    for pump in pumps.values():
        pump.soc_outer_approximation = {}
        if pump.curve_c == 1: continue

        exponent = 1 / Fraction(str(pump.curve_c))
        largest_rational, lhs_cons = obtain_soc_lhs(cone_budget, exponent)

        if largest_rational == 0:
            single_head_gain_soc({pump.label: pump}, time_steps)
            continue

        for t in range(1, time_steps + 1):
            if pump.status_decisions[t-1] == 0 or pump.fixed_flow[t-1]: continue
            z_coef = ((-pump.curve_b) ** largest_rational) * (pump.flow_upper_bound[t-1] ** ((largest_rational / exponent) - 1))
            pump.soc_outer_approximation[t] = {'method': 'multiple', 'lhs_cons': lhs_cons, 'exponent': largest_rational, 'x_coef': 1.0, 'y_coef': 1.0, 'z_coef': z_coef, 'x_bounds': [min(0.0, pump.start_node.head_lower_bound[t-1] - pump.end_node.head_upper_bound[t-1] + pump.curve_a), max(0.0, pump.start_node.head_upper_bound[t-1] - pump.end_node.head_lower_bound[t-1] + pump.curve_a)], 'y_bounds': [0.0, 1.0], 'z_bounds': [pump.positive_flow_lower_bound[t-1], pump.flow_upper_bound[t-1]]}
