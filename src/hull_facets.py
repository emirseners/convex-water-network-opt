import numpy as np
import cdd.gmp as cdd
from fractions import Fraction

def rational(value):
    return Fraction(float(value))

def box_redundant(row, lower, upper, equality_flag):
    smallest = row[0] + sum(coefficient * (lower[position] if coefficient > 0 else upper[position]) for position, coefficient in enumerate(row[1:]))
    if not equality_flag and smallest >= 0:
        return True

    largest = row[0] + sum(coefficient * (upper[position] if coefficient > 0 else lower[position]) for position, coefficient in enumerate(row[1:]))
    negligible = Fraction(1, 10 ** 11) * (largest - smallest)
    return (smallest >= -negligible and largest <= negligible) if equality_flag else (smallest >= -negligible)

def scale_coefficients(a, b, c, d):
    coeffs = np.array([a, b, c, d], dtype=float)
    magnitudes = np.abs(coeffs[coeffs != 0])
    geometric_mean = np.sqrt(magnitudes.min() * magnitudes.max())
    coeffs /= geometric_mean
    return coeffs[0], coeffs[1], coeffs[2], coeffs[3]

def find_facets(extreme_pts, box, neg=False):
    if not extreme_pts:
        return []

    lower, upper = box[0::2], box[1::2]

    generators = cdd.matrix_from_array([[Fraction(1), *point] for point in sorted(extreme_pts)], rep_type=cdd.RepType.GENERATOR)
    polyhedron = cdd.polyhedron_from_matrix(generators)
    h_representation = cdd.copy_inequalities(polyhedron)
    cdd.matrix_canonicalize(h_representation)

    equalities, candidates = [], []
    for index, row in enumerate(h_representation.array):
        equality_flag = index in h_representation.lin_set
        if box_redundant(row, lower, upper, equality_flag):
            continue
        (equalities if equality_flag else candidates).append(row)

    facets = []
    for row, equality_flag in [(row, True) for row in equalities] + [(row, False) for row in candidates]:
        coefficients = np.asarray([float(value) for value in row[1:]], dtype=float)
        norm = np.linalg.norm(coefficients)
        if norm == 0.0:
            continue

        coefficients /= norm
        d = -float(row[0]) / norm

        a, b, c = coefficients
        if neg:
            a, b, c = b, a, -c

        a, b, c, d = scale_coefficients(a, b, c, d)
        facets.append((a, b, c, d, "eq" if equality_flag else "ge"))

    return facets

def cached_facets(cache, facet_function, *arguments):
    if arguments not in cache:
        cache[arguments] = facet_function(*arguments)
    return cache[arguments]

def add_point(extreme_pts, y1, y2, y3):
    extreme_pts.add((y1, y2, y3))

def hydraulic_loss_hull_facets(y1_l, y1_u, y2_l, y2_u, y3_l, y3_u, beta, kappa, neg = False):
    loss = {rational(y3): rational(beta * (float(y3) ** kappa)) for y3 in (y3_l, y3_u)}
    beta = rational(beta)
    y1_l, y1_u, y2_l, y2_u, y3_l, y3_u = (rational(value) for value in (y1_l, y1_u, y2_l, y2_u, y3_l, y3_u))

    extreme_pts = set()
    for y2 in [y2_l, y2_u]:
        for y3 in [y3_l, y3_u]:
            y1 = min(y1_u, loss[y3] + y2)
            if y1_l <= y1:
                add_point(extreme_pts, y1, y2, y3)
                add_point(extreme_pts, y1_l, y2, y3)

    for y1 in [y1_l, y1_u]:
        for y3 in [y3_l, y3_u]:
            y2 = max(y2_l, y1 - loss[y3])
            if y2 <= y2_u:
                add_point(extreme_pts, y1, y2, y3)
                add_point(extreme_pts, y1, y2_u, y3)

    for y1 in [y1_l, y1_u]:
        for y2 in [y2_l, y2_u]:
            ratio = (y1 - y2) / beta
            y3 = max(y3_l, rational(max(0, float(ratio)) ** (1 / kappa)))
            if y3 <= y3_u:
                add_point(extreme_pts, y1, y2, y3)
                add_point(extreme_pts, y1, y2, y3_u)

    return find_facets(extreme_pts, (y1_l, y1_u, y2_l, y2_u, y3_l, y3_u), neg)

def head_gain_hull_facets(y1_l, y1_u, y2_l, y2_u, y3_l, y3_u, a, b, c):
    gain = {rational(y3): rational(a) - rational(b * (float(y3) ** c)) for y3 in (y3_l, y3_u)}
    a, b = rational(a), rational(b)
    y1_l, y1_u, y2_l, y2_u, y3_l, y3_u = (rational(value) for value in (y1_l, y1_u, y2_l, y2_u, y3_l, y3_u))

    extreme_pts = set()
    for y2 in [y2_l, y2_u]:
        for y3 in [y3_l, y3_u]:
            y1 = min(y1_u, y2 - gain[y3])
            if y1_l <= y1:
                add_point(extreme_pts, y1, y2, y3)
                add_point(extreme_pts, y1_l, y2, y3)

    for y1 in [y1_l, y1_u]:
        for y3 in [y3_l, y3_u]:
            y2 = max(y2_l, y1 + gain[y3])
            if y2 <= y2_u:
                add_point(extreme_pts, y1, y2, y3)
                add_point(extreme_pts, y1, y2_u, y3)

    for y1 in [y1_l, y1_u]:
        for y2 in [y2_l, y2_u]:
            ratio = (a + y1 - y2) / b
            y3 = max(y3_l, rational(max(0, float(ratio)) ** (1 / c)))
            if y3 <= y3_u:
                add_point(extreme_pts, y1, y2, y3)
                add_point(extreme_pts, y1, y2, y3_u)

    return find_facets(extreme_pts, (y1_l, y1_u, y2_l, y2_u, y3_l, y3_u))

def find_hydraulic_loss_facets(links, time_steps):
    cache = {}
    for link in links.values():
        link.hull_facets = {}
        for t in range(time_steps):
            if link.fixed_flow[t] or (hasattr(link, 'status_decisions') and link.status_decisions[t] == 0) or (hasattr(link, 'selection_decision') and link.selection_decision == 0): continue
            if (hasattr(link, 'direction_decision') and link.direction_decision[t] != 0) or not hasattr(link, 'direction_decision'):
                link.hull_facets[t, 'pos'] = cached_facets(cache, hydraulic_loss_hull_facets, link.start_node.head_lower_bound[t], link.start_node.head_upper_bound[t], link.end_node.head_lower_bound[t], link.end_node.head_upper_bound[t], link.positive_flow_lower_bound[t], link.flow_upper_bound[t], link.resistance_coefficient, link.exponent, False)
            if hasattr(link, 'direction_decision') and link.direction_decision[t] != 1:
                link.hull_facets[t, 'neg'] = cached_facets(cache, hydraulic_loss_hull_facets, link.end_node.head_lower_bound[t], link.end_node.head_upper_bound[t], link.start_node.head_lower_bound[t], link.start_node.head_upper_bound[t], -link.negative_flow_upper_bound[t], -link.flow_lower_bound[t], link.resistance_coefficient, link.exponent, True)

def find_head_gain_facets(pumps, time_steps):
    cache = {}
    for pump in pumps.values():
        pump.hull_facets = {}
        for t in range(time_steps):
            if pump.status_decisions[t] == 0 or pump.fixed_flow[t] or pump.curve_c == 1: continue
            if pump.flow_upper_bound[t] > 0.0:
                pump.hull_facets[t] = cached_facets(cache, head_gain_hull_facets, pump.start_node.head_lower_bound[t], pump.start_node.head_upper_bound[t], pump.end_node.head_lower_bound[t], pump.end_node.head_upper_bound[t], pump.positive_flow_lower_bound[t], pump.flow_upper_bound[t], pump.curve_a, -pump.curve_b, pump.curve_c)