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

        facets.append((a, b, c, d, "eq" if equality_flag else "ge"))

    return facets

def cached_facets(cache, facet_function, *arguments):
    if arguments not in cache:
        cache[arguments] = facet_function(*arguments)
    return cache[arguments]

def add_point(extreme_pts, y1, y2, y3):
    extreme_pts.add((y1, y2, y3))

def operational_pipe_hull_facets(y1_l, y1_u, y2_l, y2_u, y3_l, y3_u, beta, kappa, neg = False):
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

def operational_pump_hull_facets(y1_l, y1_u, y2_l, y2_u, y3_l, y3_u, a, b, c):
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

def find_pipe_facets(pipes, time_steps, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    cache = {}

    for pipe in pipes.values():
        pipe.hull_facets = {}
        for t in range(time_steps):
            if pipe.flow_lower_bound[t] == pipe.flow_upper_bound[t]: continue
            if pipe.flow_upper_bound[t] > 0:
                pipe.hull_facets[t, 'pos'] = cached_facets(cache, operational_pipe_hull_facets, pipe.start_node.head_lower_bound[t], pipe.start_node.head_upper_bound[t], pipe.end_node.head_lower_bound[t], pipe.end_node.head_upper_bound[t], max(0, pipe.flow_lower_bound[t]), pipe.flow_upper_bound[t], pipe.loss_coefficient, exponent, False)
            if pipe.flow_lower_bound[t] < 0:
                pipe.hull_facets[t, 'neg'] = cached_facets(cache, operational_pipe_hull_facets, pipe.end_node.head_lower_bound[t], pipe.end_node.head_upper_bound[t], pipe.start_node.head_lower_bound[t], pipe.start_node.head_upper_bound[t], -min(0, pipe.flow_upper_bound[t]), -pipe.flow_lower_bound[t], pipe.loss_coefficient, exponent, True)

def find_pump_facets(pumps, time_steps):
    cache = {}

    for pump in pumps.values():
        pump.hull_facets = {}
        for t in range(time_steps):
            if pump.flow_lower_bound[t] == pump.flow_upper_bound[t] or pump.curve_c == 1: continue
            if pump.flow_upper_bound[t] > 0:
                pump.hull_facets[t] = cached_facets(cache, operational_pump_hull_facets, pump.start_node.head_lower_bound[t], pump.start_node.head_upper_bound[t], pump.end_node.head_lower_bound[t], pump.end_node.head_upper_bound[t], pump.flow_lower_bound[t], pump.flow_upper_bound[t], pump.curve_a, -pump.curve_b, pump.curve_c)