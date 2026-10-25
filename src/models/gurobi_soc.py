import os
import numpy as np
from gurobipy import GRB, Model, Env, Var, quicksum

def _variable_bounds(model):
    head_lower_bounds = {(node,t): model._nodes[node].head_lower_bound[t-1] for (node, t) in model._head_indices}
    head_upper_bounds = {(node,t): model._nodes[node].head_upper_bound[t-1] for (node, t) in model._head_indices}
    flow_lower_bounds = {(link,t): model._links[link].flow_lower_bound[t-1] if link not in model._set_of_pipe_families else min((pipe.flow_lower_bound[t-1] for pipe in model._pipe_families[link].options if pipe.selection_decision != 0), default=0.0) for link in model._set_of_links for t in model._set_of_time_periods}
    flow_upper_bounds = {(link,t): model._links[link].flow_upper_bound[t-1] if link not in model._set_of_pipe_families else max((pipe.flow_upper_bound[t-1] for pipe in model._pipe_families[link].options if pipe.selection_decision != 0), default=0.0) for link in model._set_of_links for t in model._set_of_time_periods}
    head_start_lower_bounds = {(arc,t): min(0.0, model._arcs[arc].start_node.head_lower_bound[t-1]) for arc in model._set_of_arcs for t in model._set_of_time_periods}
    head_start_upper_bounds = {(arc,t): max(0.0, model._arcs[arc].start_node.head_upper_bound[t-1]) for arc in model._set_of_arcs for t in model._set_of_time_periods}
    head_end_lower_bounds = {(arc,t): min(0.0, model._arcs[arc].end_node.head_lower_bound[t-1]) for arc in model._set_of_arcs for t in model._set_of_time_periods}
    head_end_upper_bounds = {(arc,t): max(0.0, model._arcs[arc].end_node.head_upper_bound[t-1]) for arc in model._set_of_arcs for t in model._set_of_time_periods}
    return head_lower_bounds, head_upper_bounds, flow_lower_bounds, flow_upper_bounds, head_start_lower_bounds, head_start_upper_bounds, head_end_lower_bounds, head_end_upper_bounds

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_non_reservoir_junctions = [label for label, junction in model._junctions.items() if junction.segment != 'reservoir']
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_time_periods = list(range(1, model._time_steps + 1))
    model._set_of_time_periods_with_terminal = list(range(1, model._time_steps + 2))
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_arcs = list(model._arcs.keys())
    model._set_of_links = list(model._links.keys())
    model._set_of_pipe_families = list(model._pipe_families.keys())
    model._incoming_edges = {node: [] for node in model._set_of_nodes}
    model._outgoing_edges = {node: [] for node in model._set_of_nodes}

    for label, link in model._links.items():
        model._incoming_edges[link.end_node.label].append(label)
        model._outgoing_edges[link.start_node.label].append(label)

def DefineDecisionVariables(model):
    model._head_indices = [(node,t) for node in model._set_of_nodes for t in model._set_of_time_periods] + [(tank, model._time_steps + 1) for tank in model._set_of_tanks]
    head_lb, head_ub, flow_lb, flow_ub, head_start_lb, head_start_ub, head_end_lb, head_end_ub = _variable_bounds(model)

    model._head = model.addVars(model._head_indices, vtype=GRB.CONTINUOUS, lb=head_lb, ub=head_ub, name="head")
    model._flow = model.addVars([(link,t) for link in model._set_of_links for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=flow_lb, ub=flow_ub, name="flow")
    model._pump_status = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.BINARY if model._binary_pump_valve else GRB.CONTINUOUS, lb={(pump,t): 1 if model._pumps[pump].status_decisions[t-1] == 1 else 0 for pump in model._set_of_pumps for t in model._set_of_time_periods}, ub={(pump,t): 0 if model._pumps[pump].status_decisions[t-1] == 0 else 1 for pump in model._set_of_pumps for t in model._set_of_time_periods}, name="pump_status")
    model._valve_status = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.BINARY if model._binary_pump_valve else GRB.CONTINUOUS, lb={(valve,t): 1 if model._valves[valve].status_decisions[t-1] == 1 else 0 for valve in model._set_of_valves for t in model._set_of_time_periods}, ub={(valve,t): 0 if model._valves[valve].status_decisions[t-1] == 0 else 1 for valve in model._set_of_valves for t in model._set_of_time_periods}, name="valve_status")
    model._flow_status_pos = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.BINARY if model._binary_flow_direction else GRB.CONTINUOUS, lb=0.0, ub={(pipe,t): 0 if model._pipes[pipe].selection_decision == 0 or model._pipes[pipe].direction_decision[t-1] == 0 else 1 for pipe in model._set_of_pipes for t in model._set_of_time_periods}, name="flow_status_pos")
    model._flow_status_neg = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.BINARY if model._binary_flow_direction else GRB.CONTINUOUS, lb=0.0, ub={(pipe,t): 0 if model._pipes[pipe].selection_decision == 0 or model._pipes[pipe].direction_decision[t-1] == 1 else 1 for pipe in model._set_of_pipes for t in model._set_of_time_periods}, name="flow_status_neg")

    if model._problem == 'operational':
        model._pipe_status = {pipe: 1.0 for pipe in model._set_of_pipes}
    else:
        model._pipe_status = model.addVars(model._set_of_pipes, vtype=GRB.BINARY if model._binary_pipe_selection else GRB.CONTINUOUS, lb={pipe: 1 if model._pipes[pipe].selection_decision == 1 else 0 for pipe in model._set_of_pipes}, ub={pipe: 0 if model._pipes[pipe].selection_decision == 0 else 1 for pipe in model._set_of_pipes}, name="pipe_status")

    positive_indices = [(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods if model._pipes[pipe].selection_decision != 0 and model._pipes[pipe].direction_decision[t-1] != 0]
    negative_indices = [(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods if model._pipes[pipe].selection_decision != 0 and model._pipes[pipe].direction_decision[t-1] != 1]
    model._flow_pos = model.addVars([(pipe,t) for pipe,t in positive_indices if not model._pipes[pipe].fixed_flow[t-1] and model._pipes[pipe].positive_flow_lower_bound[t-1] != model._pipes[pipe].flow_upper_bound[t-1]], lb=0.0, ub={(pipe,t): max(0.0, model._pipes[pipe].flow_upper_bound[t-1]) for pipe,t in positive_indices}, name="flow_pos")
    model._flow_neg = model.addVars([(pipe,t) for pipe,t in negative_indices if not model._pipes[pipe].fixed_flow[t-1] and model._pipes[pipe].flow_lower_bound[t-1] != model._pipes[pipe].negative_flow_upper_bound[t-1]], lb={(pipe,t): min(0.0, model._pipes[pipe].flow_lower_bound[t-1]) for pipe,t in negative_indices}, ub=0.0, name="flow_neg")

    positive_indices += [(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods if model._valves[valve].status_decisions[t-1] != 0 and not model._valves[valve].fixed_flow[t-1]]
    positive_indices += [(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods if model._pumps[pump].status_decisions[t-1] != 0 and not model._pumps[pump].fixed_flow[t-1] and model._pumps[pump].curve_c != 1.0]
    model._head_start_pos = model.addVars(positive_indices, lb=head_start_lb, ub=head_start_ub, name="head_start_pos")
    model._head_start_neg = model.addVars(negative_indices, lb=head_start_lb, ub=head_start_ub, name="head_start_neg")
    model._head_end_pos = model.addVars(positive_indices, lb=head_end_lb, ub=head_end_ub, name="head_end_pos")
    model._head_end_neg = model.addVars(negative_indices, lb=head_end_lb, ub=head_end_ub, name="head_end_neg")

def DefineObjective(model):
    model.setObjective(quicksum(model._pumps[pump].fixed_cost[t-1] * model._pump_status[pump,t] + model._pumps[pump].operational_cost[t-1] * model._flow[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods) + (quicksum(model._pipes[pipe].installation_cost * model._pipe_status[pipe] for pipe in model._set_of_pipes if model._pipes[pipe].installation_cost) if model._problem != 'operational' else 0.0), GRB.MINIMIZE)

def AddPipeFacets(model):
    model._pipe_facet_cons = []
    for pipe in model._pipes.values():
        for (t, orthant), facets in pipe.hull_facets.items():
            if (pipe.label,t+1) not in (model._flow_pos if orthant == 'pos' else model._flow_neg): continue
            for a, b, c, d, sense in facets:
                if sense == 'ge':
                    model._pipe_facet_cons.append(model.addConstr(a * (model._head_start_pos[pipe.label,t+1] if orthant == 'pos' else model._head_start_neg[pipe.label,t+1]) + b * (model._head_end_pos[pipe.label,t+1] if orthant == 'pos' else model._head_end_neg[pipe.label,t+1]) + c * (model._flow_pos[pipe.label,t+1] if orthant == 'pos' else model._flow_neg[pipe.label,t+1]) >= d * (model._flow_status_pos[pipe.label,t+1] if orthant == 'pos' else model._flow_status_neg[pipe.label,t+1])))
                elif sense == 'le':
                    model._pipe_facet_cons.append(model.addConstr(a * (model._head_start_pos[pipe.label,t+1] if orthant == 'pos' else model._head_start_neg[pipe.label,t+1]) + b * (model._head_end_pos[pipe.label,t+1] if orthant == 'pos' else model._head_end_neg[pipe.label,t+1]) + c * (model._flow_pos[pipe.label,t+1] if orthant == 'pos' else model._flow_neg[pipe.label,t+1]) <= d * (model._flow_status_pos[pipe.label,t+1] if orthant == 'pos' else model._flow_status_neg[pipe.label,t+1])))
                else:
                    model._pipe_facet_cons.append(model.addConstr(a * (model._head_start_pos[pipe.label,t+1] if orthant == 'pos' else model._head_start_neg[pipe.label,t+1]) + b * (model._head_end_pos[pipe.label,t+1] if orthant == 'pos' else model._head_end_neg[pipe.label,t+1]) + c * (model._flow_pos[pipe.label,t+1] if orthant == 'pos' else model._flow_neg[pipe.label,t+1]) == d * (model._flow_status_pos[pipe.label,t+1] if orthant == 'pos' else model._flow_status_neg[pipe.label,t+1])))

def AddValveFacets(model):
    model._valve_facet_cons = []
    for valve in model._valves.values():
        for (t, orthant), facets in valve.hull_facets.items():
            if (valve.label,t+1) not in model._head_start_pos or valve.positive_flow_lower_bound[t] == valve.flow_upper_bound[t]: continue
            for a, b, c, d, sense in facets:
                if sense == 'ge':
                    model._valve_facet_cons.append(model.addConstr(a * model._head_start_pos[valve.label,t+1] + b * model._head_end_pos[valve.label,t+1] + c * model._flow[valve.label,t+1] >= d * model._valve_status[valve.label,t+1]))
                elif sense == 'le':
                    model._valve_facet_cons.append(model.addConstr(a * model._head_start_pos[valve.label,t+1] + b * model._head_end_pos[valve.label,t+1] + c * model._flow[valve.label,t+1] <= d * model._valve_status[valve.label,t+1]))
                else:
                    model._valve_facet_cons.append(model.addConstr(a * model._head_start_pos[valve.label,t+1] + b * model._head_end_pos[valve.label,t+1] + c * model._flow[valve.label,t+1] == d * model._valve_status[valve.label,t+1]))

def AddPumpFacets(model):
    model._pump_facet_cons = []
    for pump in model._pumps.values():
        for t, facets in pump.hull_facets.items():
            if (pump.label,t+1) not in model._head_start_pos or pump.positive_flow_lower_bound[t] == pump.flow_upper_bound[t]: continue
            for a, b, c, d, sense in facets:
                if sense == 'ge':
                    model._pump_facet_cons.append(model.addConstr(a * model._head_start_pos[pump.label,t+1] + b * model._head_end_pos[pump.label,t+1] + c * model._flow[pump.label,t+1] >= d * model._pump_status[pump.label,t+1]))
                elif sense == 'le':
                    model._pump_facet_cons.append(model.addConstr(a * model._head_start_pos[pump.label,t+1] + b * model._head_end_pos[pump.label,t+1] + c * model._flow[pump.label,t+1] <= d * model._pump_status[pump.label,t+1]))
                else:
                    model._pump_facet_cons.append(model.addConstr(a * model._head_start_pos[pump.label,t+1] + b * model._head_end_pos[pump.label,t+1] + c * model._flow[pump.label,t+1] == d * model._pump_status[pump.label,t+1]))

def AddMultipleSocConstraint(model, cone, x_dv, y_dv, z_dv, label=''):
    x_coef, x_bounds, y_coef, y_bounds, z_coef, z_bounds = cone['x_coef'], cone['x_bounds'], cone['y_coef'], cone['y_bounds'], cone['z_coef'], cone['z_bounds']
    model.addConstr(x_coef * x_dv >= max(0.0, x_coef * x_bounds[0]), name=f'soc_x_bound({label})')
    upper_bounds = {'x': x_coef * x_bounds[1], 'y': y_coef * y_bounds[1], 'aux_0': z_coef * max(abs(z_bounds[0]), abs(z_bounds[1]))}

    for i, j, k in cone['lhs_cons'][:-1]:
        upper_bounds[k] = (upper_bounds[i] ** 0.5) * (upper_bounds[j] ** 0.5)

    v = model.addVars(range(1, len(cone['lhs_cons'])), vtype=GRB.CONTINUOUS, lb=0.0, ub={idx: upper_bounds[f'aux_{idx}'] for idx in range(1, len(cone['lhs_cons']))}, name=f'soc_aux({label})')

    z_sign = -1.0 if z_bounds[1] <= 0.0 else 1.0
    pairs = {f'aux_{idx}': v[idx] for idx in range(1, len(cone['lhs_cons']))}
    pairs.update({'x': x_coef * x_dv, 'y': y_coef * y_dv, 'aux_0': z_sign * z_coef * z_dv})

    for idx, (i, j, k) in enumerate(cone['lhs_cons']):
        first_term = model.addVar(lb=-0.5 * upper_bounds[j], ub=0.5 * upper_bounds[i], name=f'soc_first_term({label},{idx})')
        third_term = model.addVar(lb=0.0, ub=0.5 * (upper_bounds[i] + upper_bounds[j]), name=f'soc_third_term({label},{idx})')
        model.addConstr(first_term == 0.5 * (pairs[i] - pairs[j]), name=f'soc_first_term_def({label},{idx})')
        model.addConstr(third_term == 0.5 * (pairs[i] + pairs[j]), name=f'soc_third_term_def({label},{idx})')
        model.addQConstr(first_term * first_term + pairs[k] * pairs[k] <= third_term * third_term, name=f'soc_({label},{idx})')

def AddMultipleRotatedSocConstraint(model, cone, x_dv, y_dv, z_dv, label=''):
    x_coef, x_bounds, y_coef, y_bounds, z_coef, z_bounds = cone['x_coef'], cone['x_bounds'], cone['y_coef'], cone['y_bounds'], cone['z_coef'], cone['z_bounds']
    x = model.addVar(lb=max(0.0, x_coef * x_bounds[0]), ub=x_coef * x_bounds[1], name=f'soc_x({label})')
    model.addConstr(x == x_coef * x_dv, name=f'soc_x_def({label})')

    if isinstance(y_dv, Var):
        y = y_coef * y_dv
    else:
        y = model.addVar(lb=max(0.0, y_coef * y_bounds[0]), ub=y_coef * y_bounds[1], name=f'soc_y({label})')
        model.addConstr(y == y_coef * y_dv, name=f'soc_y_def({label})')

    if z_bounds[1] <= 0.0:
        z = model.addVar(lb=0.0, ub=z_coef * max(abs(z_bounds[0]), abs(z_bounds[1])), name=f'soc_z({label})')
        model.addConstr(z == -z_coef * z_dv, name=f'soc_z_def({label})')
    else:
        z = z_coef * z_dv

    upper_bounds = {'x': x_coef * x_bounds[1], 'y': y_coef * y_bounds[1], 'aux_0': z_coef * max(abs(z_bounds[0]), abs(z_bounds[1]))}

    for i, j, k in cone['lhs_cons'][:-1]:
        upper_bounds[k] = (upper_bounds[i] ** 0.5) * (upper_bounds[j] ** 0.5)

    v = model.addVars(range(1, len(cone['lhs_cons'])), vtype=GRB.CONTINUOUS, lb=0.0, ub={idx: upper_bounds[f'aux_{idx}'] for idx in range(1, len(cone['lhs_cons']))}, name=f'soc_aux({label})')

    pairs = {f'aux_{idx}': v[idx] for idx in range(1, len(cone['lhs_cons']))}
    pairs.update({'x': x, 'y': y, 'aux_0': z})

    for idx, (i, j, k) in enumerate(cone['lhs_cons']):
        model.addQConstr(pairs[k] * pairs[k] <= pairs[i] * pairs[j], name=f'soc_({label},{idx})')

def AddSocConstraint(model, cone, x_dv, y_dv, z_dv, label=''):
    if cone['method'] == 'multiple':
        AddMultipleSocConstraint(model, cone, x_dv, y_dv, z_dv, label)
        return
    exponent = cone['exponent']
    if exponent <= 0.5:
        x_coef, x_bounds, y_coef, y_bounds, z_coef, z_bounds = cone['x_coef'], cone['x_bounds'], cone['y_coef'], cone['y_bounds'], cone['z_coef'], cone['z_bounds']
    else:
        x_dv, x_coef, x_bounds, y_dv, y_coef, y_bounds, z_coef, z_bounds, exponent = y_dv, cone['y_coef'], cone['y_bounds'], x_dv, cone['x_coef'], cone['x_bounds'], cone['z_coef'], cone['z_bounds'], 1 - exponent

    weight = ((1 + exponent) * (2 - exponent)) ** 0.5
    scale = (((1 + exponent) * (2 - exponent)) / (exponent ** 2)) ** (exponent / 2)

    x_lower, x_upper = max(0.0, x_coef * x_bounds[0]), x_coef * x_bounds[1]
    y_lower, y_upper = max(0.0, y_coef * y_bounds[0]), y_coef * y_bounds[1]

    v1 = model.addVar(lb=weight * x_lower - (1 + exponent) * y_upper, ub=weight * x_upper - (1 + exponent) * y_lower, name=f'soc_v1({label})')
    v3 = model.addVar(lb=weight * x_lower + (2 - exponent) * y_lower, ub=weight * x_upper + (2 - exponent) * y_upper, name=f'soc_v3({label})')

    model.addConstr(v1 == weight * x_coef * x_dv - (1 + exponent) * y_coef * y_dv, name=f'soc_v1_def({label})')
    model.addConstr(v3 == weight * x_coef * x_dv + (2 - exponent) * y_coef * y_dv, name=f'soc_v3_def({label})')

    model.addConstr(x_coef * x_dv >= x_lower, name=f'soc_x_bound({label})')
    model.addConstr(y_coef * y_dv >= y_lower, name=f'soc_y_bound({label})')

    v2 = (3 ** 0.5) * scale * z_coef * z_dv

    model.addQConstr(v1 * v1 + v2 * v2 <= v3 * v3, name=f'soc_({label})')

def AddRotatedSocConstraint(model, cone, x_dv, y_dv, z_dv, label=''):
    if cone['method'] == 'multiple':
        AddMultipleRotatedSocConstraint(model, cone, x_dv, y_dv, z_dv, label)
        return
    exponent = cone['exponent']
    if exponent <= 0.5:
        x_coef, x_bounds, y_coef, y_bounds, z_coef, z_bounds = cone['x_coef'], cone['x_bounds'], cone['y_coef'], cone['y_bounds'], cone['z_coef'], cone['z_bounds']
    else:
        x_dv, x_coef, x_bounds, y_dv, y_coef, y_bounds, z_coef, z_bounds, exponent = y_dv, cone['y_coef'], cone['y_bounds'], x_dv, cone['x_coef'], cone['x_bounds'], cone['z_coef'], cone['z_bounds'], 1 - exponent

    weight = ((1 + exponent) * (2 - exponent)) ** 0.5
    scale = (((1 + exponent) * (2 - exponent)) / (exponent ** 2)) ** (exponent / 2)

    if isinstance(y_dv, Var):
        y = y_coef * y_dv
    else:
        y = model.addVar(lb=max(0.0, y_coef * y_bounds[0]), ub=y_coef * y_bounds[1], name=f'soc_y({label})')
        model.addConstr(y == y_coef * y_dv, name=f'soc_y_def({label})')

    u = model.addVar(lb=0.0, ub=2 * weight * x_coef * x_bounds[1] + (1 - 2 * exponent) * y_coef * y_bounds[1], name=f'soc_u({label})')
    model.addConstr(u == 2 * weight * x_coef * x_dv + (1 - 2 * exponent) * y, name=f'soc_u_def({label})')

    model.addConstr(x_coef * x_dv >= max(0.0, x_coef * x_bounds[0]), name=f'soc_x_bound({label})')

    p = scale * z_coef * z_dv

    model.addQConstr(p * p <= y * u, name=f'soc_({label})')

def AddHeadLossConstraints(model):
    for pipe_family in model._pipe_families.values():
        for t in model._set_of_time_periods:
            model.addConstr(model._flow[pipe_family.label,t] == quicksum(pipe.flow_lower_bound[t-1] if pipe.fixed_flow[t-1] else (model._flow_pos[pipe.label,t] if (pipe.label,t) in model._flow_pos else pipe.positive_flow_lower_bound[t-1] * model._flow_status_pos[pipe.label,t]) + (model._flow_neg[pipe.label,t] if (pipe.label,t) in model._flow_neg else pipe.negative_flow_upper_bound[t-1] * model._flow_status_neg[pipe.label,t]) for pipe in pipe_family.options if pipe.selection_decision != 0), name=f'flow_equality({pipe_family.label},{t})')
            model.addConstr(model._head[pipe_family.start_node.label,t] - quicksum(model._head_start_pos.get((pipe.label,t), 0.0) + model._head_start_neg.get((pipe.label,t), 0.0) for pipe in pipe_family.options) >= pipe_family.start_node.head_lower_bound[t-1] * (1 - quicksum(model._pipe_status[pipe.label] for pipe in pipe_family.options)))
            model.addConstr(model._head[pipe_family.start_node.label,t] - quicksum(model._head_start_pos.get((pipe.label,t), 0.0) + model._head_start_neg.get((pipe.label,t), 0.0) for pipe in pipe_family.options) <= pipe_family.start_node.head_upper_bound[t-1] * (1 - quicksum(model._pipe_status[pipe.label] for pipe in pipe_family.options)))
            model.addConstr(model._head[pipe_family.end_node.label,t] - quicksum(model._head_end_pos.get((pipe.label,t), 0.0) + model._head_end_neg.get((pipe.label,t), 0.0) for pipe in pipe_family.options) >= pipe_family.end_node.head_lower_bound[t-1] * (1 - quicksum(model._pipe_status[pipe.label] for pipe in pipe_family.options)))
            model.addConstr(model._head[pipe_family.end_node.label,t] - quicksum(model._head_end_pos.get((pipe.label,t), 0.0) + model._head_end_neg.get((pipe.label,t), 0.0) for pipe in pipe_family.options) <= pipe_family.end_node.head_upper_bound[t-1] * (1 - quicksum(model._pipe_status[pipe.label] for pipe in pipe_family.options)))

    for pipe in model._pipes.values():
        if pipe.selection_decision == 0: continue
        for t in model._set_of_time_periods:
            model.addConstr(model._pipe_status[pipe.label] == model._flow_status_pos[pipe.label,t] + model._flow_status_neg[pipe.label,t], name=f'flow_status_equality({pipe.label},{t})')
            if (pipe.label,t) in model._head_start_pos:
                model.addConstr(model._head_start_pos[pipe.label,t] >= pipe.start_node.head_lower_bound[t-1] * model._flow_status_pos[pipe.label,t], name=f'head_start_pos_l({pipe},{t})')
                model.addConstr(model._head_start_pos[pipe.label,t] <= pipe.start_node.head_upper_bound[t-1] * model._flow_status_pos[pipe.label,t], name=f'head_start_pos_u({pipe},{t})')
                model.addConstr(model._head_end_pos[pipe.label,t] >= pipe.end_node.head_lower_bound[t-1] * model._flow_status_pos[pipe.label,t], name=f'head_end_pos_l({pipe},{t})')
                model.addConstr(model._head_end_pos[pipe.label,t] <= pipe.end_node.head_upper_bound[t-1] * model._flow_status_pos[pipe.label,t], name=f'head_end_pos_u({pipe},{t})')

                if pipe.positive_flow_lower_bound[t-1] == pipe.flow_upper_bound[t-1]:
                    model.addConstr(model._head_start_pos[pipe.label,t] - model._head_end_pos[pipe.label,t] == pipe.resistance_coefficient * (pipe.positive_flow_lower_bound[t-1] ** pipe.exponent) * model._flow_status_pos[pipe.label,t])
                else:
                    model.addConstr(model._flow_pos[pipe.label,t] >= pipe.positive_flow_lower_bound[t-1] * model._flow_status_pos[pipe.label,t], name=f'flow_pos_l({pipe},{t})')
                    model.addConstr(model._flow_pos[pipe.label,t] <= pipe.flow_upper_bound[t-1] * model._flow_status_pos[pipe.label,t], name=f'flow_pos_u({pipe},{t})')
                    AddSocConstraint(model, pipe.soc_outer_approximation[t,'pos'], model._head_start_pos[pipe.label,t] - model._head_end_pos[pipe.label,t], model._flow_status_pos[pipe.label,t], model._flow_pos[pipe.label,t], f'{pipe.label},{t},pos')

            if (pipe.label,t) in model._head_start_neg:
                model.addConstr(model._head_start_neg[pipe.label,t] >= pipe.start_node.head_lower_bound[t-1] * model._flow_status_neg[pipe.label,t], name=f'head_start_neg_l({pipe},{t})')
                model.addConstr(model._head_start_neg[pipe.label,t] <= pipe.start_node.head_upper_bound[t-1] * model._flow_status_neg[pipe.label,t], name=f'head_start_neg_u({pipe},{t})')
                model.addConstr(model._head_end_neg[pipe.label,t] >= pipe.end_node.head_lower_bound[t-1] * model._flow_status_neg[pipe.label,t], name=f'head_end_neg_l({pipe},{t})')
                model.addConstr(model._head_end_neg[pipe.label,t] <= pipe.end_node.head_upper_bound[t-1] * model._flow_status_neg[pipe.label,t], name=f'head_end_neg_u({pipe},{t})')

                if pipe.flow_lower_bound[t-1] == pipe.negative_flow_upper_bound[t-1]:
                    model.addConstr(model._head_start_neg[pipe.label,t] - model._head_end_neg[pipe.label,t] == pipe.resistance_coefficient * np.sign(pipe.flow_lower_bound[t-1]) * abs(pipe.flow_lower_bound[t-1]) ** pipe.exponent * model._flow_status_neg[pipe.label,t])
                else:
                    model.addConstr(model._flow_neg[pipe.label,t] >= pipe.flow_lower_bound[t-1] * model._flow_status_neg[pipe.label,t], name=f'flow_neg_l({pipe},{t})')
                    model.addConstr(model._flow_neg[pipe.label,t] <= pipe.negative_flow_upper_bound[t-1] * model._flow_status_neg[pipe.label,t], name=f'flow_neg_u({pipe},{t})')
                    AddSocConstraint(model, pipe.soc_outer_approximation[t,'neg'], model._head_end_neg[pipe.label,t] - model._head_start_neg[pipe.label,t], model._flow_status_neg[pipe.label,t], model._flow_neg[pipe.label,t], f'{pipe.label},{t},neg')

def AddPipeFlowConstraints(model):
    if model._problem == 'operational': return
    for pipe_family in model._set_of_pipe_families:
        if model._pipe_families[pipe_family].is_optional:
            model.addConstr(quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) <= 1, name=f'pipe_family_binary({pipe_family})')
        else:
            model.addConstr(quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) == 1, name=f'pipe_family_binary({pipe_family})')

        for t in model._set_of_time_periods:
            model.addConstr(model._flow[pipe_family,t] - quicksum(model._pipe_status[pipe.label] * pipe.flow_lower_bound[t-1] for pipe in model._pipe_families[pipe_family].options) >= 0, name=f'pipe_family_flow_l({pipe_family},{t})')
            model.addConstr(quicksum(model._pipe_status[pipe.label] * pipe.flow_upper_bound[t-1] for pipe in model._pipe_families[pipe_family].options) - model._flow[pipe_family,t] >= 0, name=f'pipe_family_flow_u({pipe_family},{t})')

def AddValveFlowConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].status_decisions[t-1] == -1:
                model.addConstr(model._flow[valve,t] - (model._valve_status[valve,t] * model._valves[valve].positive_flow_lower_bound[t-1]) >= 0, name=f'valve_flow_l({valve},{t})')
                model.addConstr((model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1]) - model._flow[valve,t] >= 0, name=f'valve_flow_u({valve},{t})')

def AddPumpFlowConstraints(model):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].status_decisions[t-1] == -1:
                model.addConstr(model._flow[pump,t] - (model._pump_status[pump,t] * model._pumps[pump].positive_flow_lower_bound[t-1]) >= 0, name=f'pump_flow_l({pump},{t})')
                model.addConstr((model._pump_status[pump,t] * model._pumps[pump].flow_upper_bound[t-1]) - model._flow[pump,t] >= 0, name=f'pump_flow_u({pump},{t})')

def AddPumpHeadConstraints(model):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].status_decisions[t-1] == 0: continue
            if model._pumps[pump].fixed_flow[t-1]:
                head_gain_flow = model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c)
            elif model._pumps[pump].curve_c == 1.0:
                head_gain_flow = model._pumps[pump].curve_b * model._flow[pump,t]
            else:
                model.addConstr(model._head_start_pos[pump,t] >= model._pumps[pump].start_node.head_lower_bound[t-1] * model._pump_status[pump,t], name=f'head_start_pos_l({pump},{t})')
                model.addConstr(model._head_start_pos[pump,t] <= model._pumps[pump].start_node.head_upper_bound[t-1] * model._pump_status[pump,t], name=f'head_start_pos_u({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head_start_pos[pump,t] >= model._pumps[pump].start_node.head_lower_bound[t-1] * (1 - model._pump_status[pump,t]), name=f'head_start_zero_l({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head_start_pos[pump,t] <= model._pumps[pump].start_node.head_upper_bound[t-1] * (1 - model._pump_status[pump,t]), name=f'head_start_zero_u({pump},{t})')
                model.addConstr(model._head_end_pos[pump,t] >= model._pumps[pump].end_node.head_lower_bound[t-1] * model._pump_status[pump,t], name=f'head_end_pos_l({pump},{t})')
                model.addConstr(model._head_end_pos[pump,t] <= model._pumps[pump].end_node.head_upper_bound[t-1] * model._pump_status[pump,t], name=f'head_end_pos_u({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].end_node.label,t] - model._head_end_pos[pump,t] >= model._pumps[pump].end_node.head_lower_bound[t-1] * (1 - model._pump_status[pump,t]), name=f'head_end_zero_l({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].end_node.label,t] - model._head_end_pos[pump,t] <= model._pumps[pump].end_node.head_upper_bound[t-1] * (1 - model._pump_status[pump,t]), name=f'head_end_zero_u({pump},{t})')
                if model._pumps[pump].positive_flow_lower_bound[t-1] == model._pumps[pump].flow_upper_bound[t-1]:
                    model.addConstr(model._head_start_pos[pump,t] - model._head_end_pos[pump,t] + (model._pumps[pump].curve_a + model._pumps[pump].curve_b * (model._pumps[pump].positive_flow_lower_bound[t-1] ** model._pumps[pump].curve_c)) * model._pump_status[pump,t] == 0, name=f'pump_fixed_head_gain({pump},{t})')
                else:
                    AddSocConstraint(model, model._pumps[pump].soc_outer_approximation[t], model._head_start_pos[pump,t] - model._head_end_pos[pump,t] + model._pumps[pump].curve_a * model._pump_status[pump,t], model._pump_status[pump,t], model._flow[pump,t], f'{pump},{t}')
                continue
            if model._pumps[pump].status_decisions[t-1] == 1:
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + head_gain_flow == 0, name=f'head_diff_eq({pump},{t})')
            else:
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] + head_gain_flow >= (model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]), name=f'head_diff_lower({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] + head_gain_flow <= (model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]), name=f'head_diff_upper({pump},{t})')

def AddValveHeadConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].status_decisions[t-1] == 0: continue
            if model._valves[valve].fixed_flow[t-1]:
                if model._valves[valve].status_decisions[t-1] == 1:
                    model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] == model._valves[valve].resistance_coefficient * (model._valves[valve].flow_lower_bound[t-1] ** model._valves[valve].exponent), name=f'valve_head_loss({valve},{t})')
                else:
                    model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - model._valves[valve].resistance_coefficient * (model._valves[valve].flow_lower_bound[t-1] ** model._valves[valve].exponent) >= (model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve,t]), name=f'valve_head_loss_l({valve},{t})')
                    model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - model._valves[valve].resistance_coefficient * (model._valves[valve].flow_lower_bound[t-1] ** model._valves[valve].exponent) <= (model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve,t]), name=f'valve_head_loss_u({valve},{t})')
            else:
                model.addConstr(model._head_start_pos[valve,t] >= model._valves[valve].start_node.head_lower_bound[t-1] * model._valve_status[valve,t], name=f'head_start_pos_l({valve},{t})')
                model.addConstr(model._head_start_pos[valve,t] <= model._valves[valve].start_node.head_upper_bound[t-1] * model._valve_status[valve,t], name=f'head_start_pos_u({valve},{t})')
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head_start_pos[valve,t] >= model._valves[valve].start_node.head_lower_bound[t-1] * (1 - model._valve_status[valve,t]), name=f'head_start_zero_l({valve},{t})')
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head_start_pos[valve,t] <= model._valves[valve].start_node.head_upper_bound[t-1] * (1 - model._valve_status[valve,t]), name=f'head_start_zero_u({valve},{t})')
                model.addConstr(model._head_end_pos[valve,t] >= model._valves[valve].end_node.head_lower_bound[t-1] * model._valve_status[valve,t], name=f'head_end_pos_l({valve},{t})')
                model.addConstr(model._head_end_pos[valve,t] <= model._valves[valve].end_node.head_upper_bound[t-1] * model._valve_status[valve,t], name=f'head_end_pos_u({valve},{t})')
                model.addConstr(model._head[model._valves[valve].end_node.label,t] - model._head_end_pos[valve,t] >= model._valves[valve].end_node.head_lower_bound[t-1] * (1 - model._valve_status[valve,t]), name=f'head_end_zero_l({valve},{t})')
                model.addConstr(model._head[model._valves[valve].end_node.label,t] - model._head_end_pos[valve,t] <= model._valves[valve].end_node.head_upper_bound[t-1] * (1 - model._valve_status[valve,t]), name=f'head_end_zero_u({valve},{t})')
                if model._valves[valve].positive_flow_lower_bound[t-1] == model._valves[valve].flow_upper_bound[t-1]:
                    model.addConstr(model._head_start_pos[valve,t] - model._head_end_pos[valve,t] == model._valves[valve].resistance_coefficient * (model._valves[valve].positive_flow_lower_bound[t-1] ** model._valves[valve].exponent) * model._valve_status[valve,t], name=f'valve_fixed_head_loss({valve},{t})')
                else:
                    AddSocConstraint(model, model._valves[valve].soc_outer_approximation[t,'pos'], model._head_start_pos[valve,t] - model._head_end_pos[valve,t], model._valve_status[valve,t], model._flow[valve,t], f'{valve},{t},valve')

def AddTankBalanceConstraints(model):
    for tank in model._set_of_tanks:
        for t in model._set_of_time_periods:
            model.addConstr(model._tanks[tank].tank_coef * (quicksum(model._flow[link,t] for link in model._incoming_edges[tank]) - quicksum(model._flow[link,t] for link in model._outgoing_edges[tank])) == model._head[tank,t+1] - model._head[tank,t], name=f'tank_balance({tank},{t})')

def AddJunctionBalanceConstraints(model):
    for junction in model._set_of_non_reservoir_junctions:
        for t in model._set_of_time_periods:
            model.addConstr(quicksum(model._flow[link,t] for link in model._incoming_edges[junction]) - quicksum(model._flow[link,t] for link in model._outgoing_edges[junction]) == model._junctions[junction].net_demand[t-1], name=f'junction_balance({junction},{t})')

def AddValidInequalities(model, valid_inequalities):
    model._valid_cons = []
    for index, (sense, terms, rhs) in enumerate(sorted(valid_inequalities)):
        expression = quicksum(float(coefficient) * (model._pipe_status[label] if variable == 'pipe_status' else getattr(model, '_' + variable)[label, period + 1]) for (variable, label, period), coefficient in terms)
        if sense == 'ge':
            constraint = expression >= float(rhs)
        elif sense == 'le':
            constraint = expression <= float(rhs)
        else:
            constraint = expression == float(rhs)
        model._valid_cons.append(model.addConstr(constraint, name=f'valid_inequality({index})'))

def AddNoGoodCutGurobi(model, previous_solution, lazy_flag=False, up_to=None):
    up_to = model._time_steps if up_to is None else up_to
    pump_incumbent = {(pump,t): int(previous_solution['pump_status'][pump][t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods if t <= up_to}
    valve_incumbent = {(valve,t): int(previous_solution['valve_status'][valve][t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods if t <= up_to}
    pipe_incumbent = {pipe: int(previous_solution['pipe_status'][pipe]) for pipe in model._set_of_pipes} if model._problem != 'operational' else {}
    terms = [(2 * value - 1) * model._pump_status[pump,t] for (pump,t), value in pump_incumbent.items()] + [(2 * value - 1) * model._valve_status[valve,t] for (valve,t), value in valve_incumbent.items()] + [(2 * value - 1) * model._pipe_status[pipe] for pipe, value in pipe_incumbent.items()]
    if not terms: return
    if lazy_flag:
        model.cbLazy(quicksum(terms) <= sum(pump_incumbent.values()) + sum(valve_incumbent.values()) + sum(pipe_incumbent.values()) - 1)
    else:
        model.addConstr(quicksum(terms) <= sum(pump_incumbent.values()) + sum(valve_incumbent.values()) + sum(pipe_incumbent.values()) - 1)

def GurobiModelSoc(model_name, network, binary_flow_direction=True, binary_pump_valve=True, binary_pipe_selection=False):
    model_env = Env(empty=True)
    model_env.start()
    model = Model(model_name, env=model_env)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps, model._problem = network.junctions, network.tanks, network.pipes, network.pumps, network.valves, network.time_steps, network.problem
    model._nodes, model._arcs, model._links, model._pipe_families = network.junctions | network.tanks, network.pipes | network.pumps | network.valves, network.pipe_families | network.pumps | network.valves, network.pipe_families
    model._binary_flow_direction, model._binary_pump_valve, model._binary_pipe_selection = binary_flow_direction, binary_pump_valve, binary_pipe_selection

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model)
    AddPipeFacets(model)
    AddPipeFlowConstraints(model)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
    AddValveHeadConstraints(model)
    AddValveFacets(model)
    AddPumpHeadConstraints(model)
    AddPumpFacets(model)
    AddTankBalanceConstraints(model)
    AddJunctionBalanceConstraints(model)
    if getattr(network, 'valid_inequalities', None):
        AddValidInequalities(model, network.valid_inequalities)

    #lp_filename=os.path.join(results_directory, f'{model_name}.lp')
    #model.write(lp_filename)
    #model.setParam('MIPFocus', 3)
    model.setParam('Threads', 12)
    model.setParam('TimeLimit', network.time_limit)
    log_file_path = os.path.join(network.results_directory, f'{model_name}_log.txt')
    model.setParam('LogFile', log_file_path)
    model.setParam("NonConvex", 1)

    return model, model_env

def SolveGurobiModelSoc(model, model_name, results_directory, compute_iss=False):
    model.optimize()
    problem_status = 'feasible' if model.SolCount > 0 else 'infeasible' if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else 'nosolution'
    lower_bound = model.ObjBound if problem_status != 'infeasible' and abs(model.ObjBound) < GRB.INFINITY else None

    if problem_status == 'infeasible' and compute_iss:
        model.computeIIS()
        iis_file_path = os.path.join(results_directory, f'IIS_{model_name}.ilp')
        model.write(iis_file_path)

    #model.write(os.path.join(results_directory, f'{model_name}_solution.sol'))

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            solution['head'] = {node: np.array([model._head[node, t].x for t in (model._set_of_time_periods if node in model._set_of_junctions else model._set_of_time_periods_with_terminal)]) for node in model._set_of_nodes}

        if len(model._set_of_links) > 0:
            solution['flow'] = {link: np.array([model._flow[link, t].x for t in model._set_of_time_periods]) for link in model._set_of_links}

        if len(model._set_of_pumps) > 0:
            solution['pump_status'] = {pump: np.array([model._pump_status[pump,t].x for t in model._set_of_time_periods]) for pump in model._set_of_pumps}
            if model._binary_pump_valve: solution['pump_status'] = {pump: np.rint(solution['pump_status'][pump]) for pump in model._set_of_pumps}

        if len(model._set_of_valves) > 0:
            solution['valve_status'] = {valve: np.array([model._valve_status[valve,t].x for t in model._set_of_time_periods]) for valve in model._set_of_valves}
            if model._binary_pump_valve: solution['valve_status'] = {valve: np.rint(solution['valve_status'][valve]) for valve in model._set_of_valves}

        if len(model._set_of_pipes) > 0:
            solution['pipe_status'] = {pipe: float(model._pipe_status[pipe].x) if model._problem != 'operational' else 1 for pipe in model._set_of_pipes}
            if model._binary_pipe_selection or model._binary_flow_direction: solution['pipe_status'] = {pipe: int(round(solution['pipe_status'][pipe])) for pipe in model._set_of_pipes}

    return problem_status, solution, lower_bound
