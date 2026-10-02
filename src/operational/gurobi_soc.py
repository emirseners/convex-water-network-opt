import os
import numpy as np
from gurobipy import GRB, Model, Env, Var, quicksum

def _variable_bounds(model):
    head_lower_bounds = {(node,t): model._nodes[node].head_lower_bound[t-1] for (node, t) in model._head_indices}
    head_upper_bounds = {(node,t): model._nodes[node].head_upper_bound[t-1] for (node, t) in model._head_indices}
    flow_lower_bounds = {(link,t): model._links[link].flow_lower_bound[t-1] for link in model._set_of_links for t in model._set_of_time_periods}
    flow_upper_bounds = {(link,t): model._links[link].flow_upper_bound[t-1] for link in model._set_of_links for t in model._set_of_time_periods}
    return head_lower_bounds, head_upper_bounds, flow_lower_bounds, flow_upper_bounds

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
    model._set_of_links = list(model._links.keys())

def DefineDecisionVariables(model, binary_pump_valve):
    model._head_indices = ([(node,t) for node in model._set_of_nodes for t in model._set_of_time_periods] + [(tank, model._time_steps + 1) for tank in model._set_of_tanks])
    head_lb, head_ub, flow_lb, flow_ub = _variable_bounds(model)

    model._head = model.addVars([(node,t) for (node,t) in model._head_indices], vtype=GRB.CONTINUOUS, lb=head_lb, ub=head_ub, name="head")
    model._flow = model.addVars([(link,t) for link in model._set_of_links for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=flow_lb, ub=flow_ub, name="flow")
    if binary_pump_valve:
        model._pump_status = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.BINARY, name="pump_status")
        model._valve_status = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.BINARY, name="valve_status")
    else:
        model._pump_status = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="pump_status")
        model._valve_status = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="valve_status")

def DefineObjective(model):
    model.setObjective(quicksum(model._pumps[pump].fixed_cost[t-1] * model._pump_status[pump,t] + model._pumps[pump].operational_cost[t-1] * model._flow[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods), GRB.MINIMIZE)

def AddPipeFacets(model):
    model._pipe_facet_cons = []
    for pipe in model._pipes.values():
        for (t, orthant), facets in pipe.hull_facets.items():
            for face in facets:
                a, b, c, d, sense = face
                if sense == 'ge':
                    model._pipe_facet_cons.append(model.addConstr(a * (model._head_pipe_start_plus[pipe.label,t+1] if orthant == 'pos' else model._head_pipe_start_minus[pipe.label,t+1]) + b * (model._head_pipe_end_plus[pipe.label,t+1] if orthant == 'pos' else model._head_pipe_end_minus[pipe.label,t+1]) + c * (model._flow_pipe_plus[pipe.label,t+1] if orthant == 'pos' else model._flow_pipe_minus[pipe.label,t+1]) >= d * (model._flow_pipe_status[pipe.label,t+1] if orthant == 'pos' else 1 - model._flow_pipe_status[pipe.label,t+1])))
                elif sense == 'le':
                    model._pipe_facet_cons.append(model.addConstr(a * (model._head_pipe_start_plus[pipe.label,t+1] if orthant == 'pos' else model._head_pipe_start_minus[pipe.label,t+1]) + b * (model._head_pipe_end_plus[pipe.label,t+1] if orthant == 'pos' else model._head_pipe_end_minus[pipe.label,t+1]) + c * (model._flow_pipe_plus[pipe.label,t+1] if orthant == 'pos' else model._flow_pipe_minus[pipe.label,t+1]) <= d * (model._flow_pipe_status[pipe.label,t+1] if orthant == 'pos' else 1 - model._flow_pipe_status[pipe.label,t+1])))
                else:
                    model._pipe_facet_cons.append(model.addConstr(a * (model._head_pipe_start_plus[pipe.label,t+1] if orthant == 'pos' else model._head_pipe_start_minus[pipe.label,t+1]) + b * (model._head_pipe_end_plus[pipe.label,t+1] if orthant == 'pos' else model._head_pipe_end_minus[pipe.label,t+1]) + c * (model._flow_pipe_plus[pipe.label,t+1] if orthant == 'pos' else model._flow_pipe_minus[pipe.label,t+1]) == d * (model._flow_pipe_status[pipe.label,t+1] if orthant == 'pos' else 1 - model._flow_pipe_status[pipe.label,t+1])))

def AddValveFacets(model):
    model._valve_facet_cons = []
    for valve in model._valves.values():
        for (t, orthant), facets in valve.hull_facets.items():
            for face in facets:
                a, b, c, d, sense = face
                if sense == 'ge':
                    model._valve_facet_cons.append(model.addConstr(a * model._head_valve_start_plus[valve.label,t+1] + b * model._head_valve_end_plus[valve.label,t+1] + c * model._flow[valve.label,t+1] >= d * model._valve_status[valve.label,t+1]))
                elif sense == 'le':
                    model._valve_facet_cons.append(model.addConstr(a * model._head_valve_start_plus[valve.label,t+1] + b * model._head_valve_end_plus[valve.label,t+1] + c * model._flow[valve.label,t+1] <= d * model._valve_status[valve.label,t+1]))
                else:
                    model._valve_facet_cons.append(model.addConstr(a * model._head_valve_start_plus[valve.label,t+1] + b * model._head_valve_end_plus[valve.label,t+1] + c * model._flow[valve.label,t+1] == d * model._valve_status[valve.label,t+1]))

def AddPumpFacets(model):
    model._pump_facet_cons = []
    for pump in model._pumps.values():
        for t, facets in pump.hull_facets.items():
            for face in facets:
                a, b, c, d, sense = face
                if sense == 'ge':
                    model._pump_facet_cons.append(model.addConstr(a * model._head_pump_start_one[pump.label,t+1] + b * model._head_pump_end_one[pump.label,t+1] + c * model._flow[pump.label,t+1] >= d * model._pump_status[pump.label,t+1]))
                elif sense == 'le':
                    model._pump_facet_cons.append(model.addConstr(a * model._head_pump_start_one[pump.label,t+1] + b * model._head_pump_end_one[pump.label,t+1] + c * model._flow[pump.label,t+1] <= d * model._pump_status[pump.label,t+1]))
                else:
                    model._pump_facet_cons.append(model.addConstr(a * model._head_pump_start_one[pump.label,t+1] + b * model._head_pump_end_one[pump.label,t+1] + c * model._flow[pump.label,t+1] == d * model._pump_status[pump.label,t+1]))

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

def _headloss_variable_bounds(model):
    flow_minus_lower_bounds = {(pipe,t): min(0.0, model._pipes[pipe].flow_lower_bound[t-1]) for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    flow_plus_upper_bounds = {(pipe,t): max(0.0, model._pipes[pipe].flow_upper_bound[t-1]) for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_start_lower_bounds =  {(pipe,t): min(0.0, model._pipes[pipe].start_node.head_lower_bound[t-1]) for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_start_upper_bounds =  {(pipe,t): max(0.0, model._pipes[pipe].start_node.head_upper_bound[t-1]) for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_end_lower_bounds =  {(pipe,t): min(0.0, model._pipes[pipe].end_node.head_lower_bound[t-1]) for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_end_upper_bounds =  {(pipe,t): max(0.0, model._pipes[pipe].end_node.head_upper_bound[t-1]) for pipe in model._set_of_pipes for t in model._set_of_time_periods}

    return flow_minus_lower_bounds, flow_plus_upper_bounds, head_start_lower_bounds, head_start_upper_bounds, head_end_lower_bounds, head_end_upper_bounds

def AddHeadLossConstraints(model, binary_pipe):
    flow_minus_lb, flow_plus_ub, head_start_lb, head_start_ub, head_end_lb, head_end_ub = _headloss_variable_bounds(model)

    model._flow_pipe_plus = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=0.0, ub=flow_plus_ub, name="flow_pipe_plus")
    model._flow_pipe_minus = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=flow_minus_lb, ub=0.0, name="flow_pipe_minus")
    model._head_pipe_start_plus = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_start_lb, ub=head_start_ub, name="head_pipe_start_plus")
    model._head_pipe_start_minus = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_start_lb, ub=head_start_ub, name="head_pipe_start_minus")
    model._head_pipe_end_plus = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_end_lb, ub=head_end_ub, name="_head_pipe_end_plus")
    model._head_pipe_end_minus = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_end_lb, ub=head_end_ub, name="_head_pipe_end_minus")

    if binary_pipe:
        model._flow_pipe_status = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.BINARY, name="flow_pipe_status")
    else:
        model._flow_pipe_status = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="flow_pipe_status")

    for pipe in model._set_of_pipes:
        for t in model._set_of_time_periods:
            if model._pipes[pipe].flow_lower_bound[t-1] == model._pipes[pipe].flow_upper_bound[t-1]:
                model.addConstr(model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == model._pipes[pipe].loss_coefficient * (-1 if model._pipes[pipe].flow_lower_bound[t-1] < 0.0 else 1) * (abs(model._pipes[pipe].flow_lower_bound[t-1]) ** model._pipes[pipe].exponent), name=f'head_loss({pipe},{t})')
                model._flow_pipe_status[pipe,t].lb = model._flow_pipe_status[pipe,t].ub = 1
                model._head_pipe_start_plus[pipe,t].lb = model._head_pipe_start_plus[pipe,t].ub = 0.0
                model._head_pipe_end_plus[pipe,t].lb = model._head_pipe_end_plus[pipe,t].ub = 0.0
                model._flow_pipe_plus[pipe,t].lb = model._flow_pipe_plus[pipe,t].ub = 0.0
                model._flow_pipe_minus[pipe,t].lb = model._flow_pipe_minus[pipe,t].ub = 0.0
                model._head_pipe_start_minus[pipe,t].lb = model._head_pipe_start_minus[pipe,t].ub = 0.0
                model._head_pipe_end_minus[pipe,t].lb = model._head_pipe_end_minus[pipe,t].ub = 0.0
                continue

            model.addConstr(model._flow[pipe,t] == model._flow_pipe_plus[pipe,t] + model._flow_pipe_minus[pipe,t], name=f'flow_equality({pipe},{t})')
            model.addConstr(model._head[model._pipes[pipe].start_node.label,t] == model._head_pipe_start_plus[pipe,t] + model._head_pipe_start_minus[pipe,t], name=f'head_start_equality({pipe},{t})')
            model.addConstr(model._head[model._pipes[pipe].end_node.label,t] == model._head_pipe_end_plus[pipe,t] + model._head_pipe_end_minus[pipe,t], name=f'head_end_equality({pipe},{t})')

            if model._pipes[pipe].flow_upper_bound[t-1] > 0.0:
                model.addConstr(model._head_pipe_start_plus[pipe,t] >= model._flow_pipe_status[pipe,t] * model._pipes[pipe].start_node.head_lower_bound[t-1], name=f'head_start_plus_l({pipe},{t})')
                model.addConstr(model._head_pipe_start_plus[pipe,t] <= model._flow_pipe_status[pipe,t] * model._pipes[pipe].start_node.head_upper_bound[t-1], name=f'head_start_plus_u({pipe},{t})')
                model.addConstr(model._head_pipe_end_plus[pipe,t] >= model._flow_pipe_status[pipe,t] * model._pipes[pipe].end_node.head_lower_bound[t-1], name=f'head_end_plus_l({pipe},{t})')
                model.addConstr(model._head_pipe_end_plus[pipe,t] <= model._flow_pipe_status[pipe,t] * model._pipes[pipe].end_node.head_upper_bound[t-1], name=f'head_end_plus_u({pipe},{t})')
                model.addConstr(model._flow_pipe_plus[pipe,t] >= model._flow_pipe_status[pipe,t] * max(0.0, model._pipes[pipe].flow_lower_bound[t-1]), name=f'flow_plus_l({pipe},{t})')
                model.addConstr(model._flow_pipe_plus[pipe,t] <= model._flow_pipe_status[pipe,t] * model._pipes[pipe].flow_upper_bound[t-1], name=f'flow_plus_u({pipe},{t})')
                AddSocConstraint(model, model._pipes[pipe].soc_outer_approximation[t,'pos'], model._head_pipe_start_plus[pipe,t] - model._head_pipe_end_plus[pipe,t], model._flow_pipe_status[pipe,t], model._flow_pipe_plus[pipe,t], f'{pipe},{t},pos')

            else:
                model._head_pipe_start_plus[pipe,t].lb = model._head_pipe_start_plus[pipe,t].ub = 0.0
                model._head_pipe_end_plus[pipe,t].lb = model._head_pipe_end_plus[pipe,t].ub = 0.0
                model._flow_pipe_status[pipe,t].lb = model._flow_pipe_status[pipe,t].ub = 0

            if model._pipes[pipe].flow_lower_bound[t-1] < 0.0:
                model.addConstr(model._head_pipe_start_minus[pipe,t] >= (1 - model._flow_pipe_status[pipe,t]) * model._pipes[pipe].start_node.head_lower_bound[t-1], name=f'head_start_minus_l({pipe},{t})')
                model.addConstr(model._head_pipe_start_minus[pipe,t] <= (1 - model._flow_pipe_status[pipe,t]) * model._pipes[pipe].start_node.head_upper_bound[t-1], name=f'head_start_minus_u({pipe},{t})')
                model.addConstr(model._head_pipe_end_minus[pipe,t] >= (1 - model._flow_pipe_status[pipe,t]) * model._pipes[pipe].end_node.head_lower_bound[t-1], name=f'head_end_minus_l({pipe},{t})')
                model.addConstr(model._head_pipe_end_minus[pipe,t] <= (1 - model._flow_pipe_status[pipe,t]) * model._pipes[pipe].end_node.head_upper_bound[t-1], name=f'head_end_minus_u({pipe},{t})')
                model.addConstr(model._flow_pipe_minus[pipe,t] >= (1 - model._flow_pipe_status[pipe,t]) * model._pipes[pipe].flow_lower_bound[t-1], name=f'flow_minus_l({pipe},{t})')
                model.addConstr(model._flow_pipe_minus[pipe,t] <= (1 - model._flow_pipe_status[pipe,t]) * min(0.0, model._pipes[pipe].flow_upper_bound[t-1]), name=f'flow_minus_u({pipe},{t})')
                AddSocConstraint(model, model._pipes[pipe].soc_outer_approximation[t,'neg'], model._head_pipe_end_minus[pipe,t] - model._head_pipe_start_minus[pipe,t], 1 - model._flow_pipe_status[pipe,t], model._flow_pipe_minus[pipe,t], f'{pipe},{t},neg')

            elif model._pipes[pipe].flow_upper_bound[t-1] > 0.0:
                model._head_pipe_start_minus[pipe,t].lb = model._head_pipe_start_minus[pipe,t].ub = 0.0
                model._head_pipe_end_minus[pipe,t].lb = model._head_pipe_end_minus[pipe,t].ub = 0.0
                model._flow_pipe_status[pipe,t].lb = model._flow_pipe_status[pipe,t].ub = 1

def AddValveFlowConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].decisions[t-1] == 1:
                model._valve_status[valve,t].lb = model._valve_status[valve,t].ub = 1
            elif model._valves[valve].decisions[t-1] == 0:
                model._valve_status[valve,t].lb = model._valve_status[valve,t].ub = 0
                model._flow[valve,t].lb = model._flow[valve,t].ub = 0.0
            else:
                model.addConstr(model._flow[valve,t] - (model._valve_status[valve,t] * model._valves[valve].flow_lower_bound[t-1]) >= 0, name=f'valve_flow_l({valve},{t})')
                model.addConstr((model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1]) - model._flow[valve,t] >= 0, name=f'valve_flow_u({valve},{t})')

def _pumphead_variable_bounds(model):
    head_start_lower_bounds =  {(pump,t): min(0.0, model._pumps[pump].start_node.head_lower_bound[t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods}
    head_start_upper_bounds =  {(pump,t): max(0.0, model._pumps[pump].start_node.head_upper_bound[t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods}
    head_end_lower_bounds =  {(pump,t): min(0.0, model._pumps[pump].end_node.head_lower_bound[t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods}
    head_end_upper_bounds =  {(pump,t): max(0.0, model._pumps[pump].end_node.head_upper_bound[t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods}

    return head_start_lower_bounds, head_start_upper_bounds, head_end_lower_bounds, head_end_upper_bounds

def AddPumpHeadConstraints(model):
    head_start_lb, head_start_ub, head_end_lb, head_end_ub = _pumphead_variable_bounds(model)

    model._head_pump_start_one = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_start_lb, ub=head_start_ub, name="head_pump_start_one")
    model._head_pump_end_one = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_end_lb, ub=head_end_ub, name="head_pump_end_one")

    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].decisions[t-1] == 0:
                model._flow[pump,t].lb = model._flow[pump,t].ub = 0.0
                model._pump_status[pump,t].lb = model._pump_status[pump,t].ub = 0
                model._head_pump_start_one[pump,t].lb = model._head_pump_start_one[pump,t].ub = 0.0
                model._head_pump_end_one[pump,t].lb = model._head_pump_end_one[pump,t].ub = 0.0
                continue

            if model._pumps[pump].flow_lower_bound[t-1] == model._pumps[pump].flow_upper_bound[t-1] and model._pumps[pump].flow_upper_bound[t-1] > 0.0:
                model._pump_status[pump,t].lb = model._pump_status[pump,t].ub = 1
                model._head_pump_start_one[pump,t].lb = model._head_pump_start_one[pump,t].ub = 0.0
                model._head_pump_end_one[pump,t].lb = model._head_pump_end_one[pump,t].ub = 0.0
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + (model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c)) == 0, name=f'head_gain_({pump},{t})')
                continue

            if model._pumps[pump].decisions[t-1] == 1:
                model._pump_status[pump,t].lb = model._pump_status[pump,t].ub = 1

            if model._pumps[pump].flow_lower_bound[t-1] == model._pumps[pump].flow_upper_bound[t-1] == 0.0:
                model._head_pump_start_one[pump,t].lb = model._head_pump_start_one[pump,t].ub = 0.0
                model._head_pump_end_one[pump,t].lb = model._head_pump_end_one[pump,t].ub = 0.0
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] >= model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1] - model._pumps[pump].start_node.head_lower_bound[t-1] * model._pump_status[pump,t] + model._pumps[pump].end_node.head_upper_bound[t-1] * model._pump_status[pump,t], name=f'head_gain_l_({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] <= model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1] - model._pumps[pump].start_node.head_upper_bound[t-1] * model._pump_status[pump,t] + model._pumps[pump].end_node.head_lower_bound[t-1] * model._pump_status[pump,t], name=f'head_gain_u_({pump},{t})')
                continue

            if model._pumps[pump].curve_c == 1:
                model._head_pump_start_one[pump,t].lb = model._head_pump_start_one[pump,t].ub = 0.0
                model._head_pump_end_one[pump,t].lb = model._head_pump_end_one[pump,t].ub = 0.0
                model.addConstr(model._flow[pump,t] >= model._pump_status[pump,t] * model._pumps[pump].flow_lower_bound[t-1], name=f'flow_pump_l_({pump},{t})')
                model.addConstr(model._flow[pump,t] <= model._pump_status[pump,t] * model._pumps[pump].flow_upper_bound[t-1], name=f'flow_pump_u_({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] + model._pumps[pump].curve_b * model._flow[pump,t] >= model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1] - model._pumps[pump].start_node.head_lower_bound[t-1] * model._pump_status[pump,t] + model._pumps[pump].end_node.head_upper_bound[t-1] * model._pump_status[pump,t], name=f'head_gain_l_({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] + model._pumps[pump].curve_b * model._flow[pump,t] <= model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1] - model._pumps[pump].start_node.head_upper_bound[t-1] * model._pump_status[pump,t] + model._pumps[pump].end_node.head_lower_bound[t-1] * model._pump_status[pump,t], name=f'head_gain_u_({pump},{t})')                
                continue

            model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head_pump_start_one[pump,t] >= (1 - model._pump_status[pump,t]) * model._pumps[pump].start_node.head_lower_bound[t-1], name=f'head_pump_start_zero_l_({pump},{t})')
            model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head_pump_start_one[pump,t] <= (1 - model._pump_status[pump,t]) * model._pumps[pump].start_node.head_upper_bound[t-1], name=f'head_pump_start_zero_u_({pump},{t})')
            model.addConstr(model._head_pump_start_one[pump,t] >= model._pump_status[pump,t] * model._pumps[pump].start_node.head_lower_bound[t-1], name=f'head_pump_start_one_l_({pump},{t})')
            model.addConstr(model._head_pump_start_one[pump,t] <= model._pump_status[pump,t] * model._pumps[pump].start_node.head_upper_bound[t-1], name=f'head_pump_start_one_u_({pump},{t})')
            model.addConstr(model._head[model._pumps[pump].end_node.label,t] - model._head_pump_end_one[pump,t] >= (1 - model._pump_status[pump,t]) * model._pumps[pump].end_node.head_lower_bound[t-1], name=f'head_pump_end_zero_l_({pump},{t})')
            model.addConstr(model._head[model._pumps[pump].end_node.label,t] - model._head_pump_end_one[pump,t] <= (1 - model._pump_status[pump,t]) * model._pumps[pump].end_node.head_upper_bound[t-1], name=f'head_pump_end_zero_u_({pump},{t})')
            model.addConstr(model._head_pump_end_one[pump,t] >= model._pump_status[pump,t] * model._pumps[pump].end_node.head_lower_bound[t-1], name=f'head_pump_end_one_l_({pump},{t})')
            model.addConstr(model._head_pump_end_one[pump,t] <= model._pump_status[pump,t] * model._pumps[pump].end_node.head_upper_bound[t-1], name=f'head_pump_end_one_u_({pump},{t})')
            model.addConstr(model._flow[pump,t] >= model._pump_status[pump,t] * model._pumps[pump].flow_lower_bound[t-1], name=f'flow_pump_l_({pump},{t})')
            model.addConstr(model._flow[pump,t] <= model._pump_status[pump,t] * model._pumps[pump].flow_upper_bound[t-1], name=f'flow_pump_u_({pump},{t})')
            AddSocConstraint(model, model._pumps[pump].soc_outer_approximation[t], model._head_pump_start_one[pump,t] - model._head_pump_end_one[pump,t] + model._pumps[pump].curve_a * model._pump_status[pump,t], model._pump_status[pump,t], model._flow[pump,t], f'{pump},{t}')

def AddValveHeadConstraints(model):
    head_start_lower_bounds = {(valve,t): min(0.0, model._valves[valve].start_node.head_lower_bound[t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods}
    head_start_upper_bounds = {(valve,t): max(0.0, model._valves[valve].start_node.head_upper_bound[t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods}
    head_end_lower_bounds = {(valve,t): min(0.0, model._valves[valve].end_node.head_lower_bound[t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods}
    head_end_upper_bounds = {(valve,t): max(0.0, model._valves[valve].end_node.head_upper_bound[t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods}

    model._head_valve_start_plus = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_start_lower_bounds, ub=head_start_upper_bounds, name="head_valve_start_plus")
    model._head_valve_end_plus = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_end_lower_bounds, ub=head_end_upper_bounds, name="head_valve_end_plus")

    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            model.addConstr((model._head[model._valves[valve].start_node.label,t] - model._head_valve_start_plus[valve,t]) >= (1 - model._valve_status[valve,t]) * model._valves[valve].start_node.head_lower_bound[t-1], name=f'valve_head_start_zero_l({valve},{t})')
            model.addConstr((model._head[model._valves[valve].start_node.label,t] - model._head_valve_start_plus[valve,t]) <= (1 - model._valve_status[valve,t]) * model._valves[valve].start_node.head_upper_bound[t-1], name=f'valve_head_start_zero_u({valve},{t})')
            model.addConstr((model._head[model._valves[valve].end_node.label,t] - model._head_valve_end_plus[valve,t]) >= (1 - model._valve_status[valve,t]) * model._valves[valve].end_node.head_lower_bound[t-1], name=f'valve_head_end_zero_l({valve},{t})')
            model.addConstr((model._head[model._valves[valve].end_node.label,t] - model._head_valve_end_plus[valve,t]) <= (1 - model._valve_status[valve,t]) * model._valves[valve].end_node.head_upper_bound[t-1], name=f'valve_head_end_zero_u({valve},{t})')

            model.addConstr(model._head_valve_start_plus[valve,t] >= model._valve_status[valve,t] * model._valves[valve].start_node.head_lower_bound[t-1], name=f'valve_head_start_plus_l({valve},{t})')
            model.addConstr(model._head_valve_start_plus[valve,t] <= model._valve_status[valve,t] * model._valves[valve].start_node.head_upper_bound[t-1], name=f'valve_head_start_plus_u({valve},{t})')
            model.addConstr(model._head_valve_end_plus[valve,t] >= model._valve_status[valve,t] * model._valves[valve].end_node.head_lower_bound[t-1], name=f'valve_head_end_plus_l({valve},{t})')
            model.addConstr(model._head_valve_end_plus[valve,t] <= model._valve_status[valve,t] * model._valves[valve].end_node.head_upper_bound[t-1], name=f'valve_head_end_plus_u({valve},{t})')

            if model._valves[valve].flow_lower_bound[t-1] == model._valves[valve].flow_upper_bound[t-1]:
                model.addConstr(model._head_valve_start_plus[valve,t] - model._head_valve_end_plus[valve,t] == model._valves[valve].loss_coefficient * (model._valves[valve].flow_lower_bound[t-1] ** model._valves[valve].exponent) * model._valve_status[valve,t], name=f'valve_fixed_head_loss({valve},{t})')
            elif model._valves[valve].decisions[t-1] != 0:
                AddSocConstraint(model, model._valves[valve].soc_outer_approximation[t,'pos'], model._head_valve_start_plus[valve,t] - model._head_valve_end_plus[valve,t], model._valve_status[valve,t], model._flow[valve,t], f'{valve},{t},valve')

def AddTankBalanceConstraints(model, hydraulic_timestep):
    for tank in model._set_of_tanks:
        for t in model._set_of_time_periods:
            model.addConstr((hydraulic_timestep / model._tanks[tank].area) * quicksum(model._flow[edge.label,t] for edge in model._tanks[tank].incoming_edges) - (hydraulic_timestep / model._tanks[tank].area) * quicksum(model._flow[edge.label,t] for edge in model._tanks[tank].outgoing_edges) == model._head[tank,t+1] - model._head[tank,t], name=f'tank_balance({tank},{t})')

def AddJunctionBalanceConstraints(model):
    for junction in model._set_of_non_reservoir_junctions:
        for t in model._set_of_time_periods:
            model.addConstr(quicksum(model._flow[edge.label,t] for edge in model._junctions[junction].incoming_edges) - quicksum(model._flow[edge.label,t] for edge in model._junctions[junction].outgoing_edges) == model._junctions[junction].net_demand[t-1], name=f'junction_balance({junction},{t})')

def AddValidInequalities(model, valid_inequalities):
    model._valid_cons = []
    for index, (sense, terms, rhs) in enumerate(sorted(valid_inequalities)):
        expression = quicksum(float(coefficient) * getattr(model, '_' + variable)[label, period + 1] for (variable, label, period), coefficient in terms)
        if sense == 'ge':
            constraint = expression >= float(rhs)
        else:
            constraint = expression == float(rhs)
        model._valid_cons.append(model.addConstr(constraint, name=f'valid_inequality({index})'))

def AddNoGoodCutSoc(model, previous_solution, lazy_flag=False, up_to=None):
    up_to = model._time_steps if up_to == None else up_to
    pump_incumbent = {(pump, t): np.array(previous_solution["pump_status"][pump][t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods if t <= up_to}
    valve_incumbent = {(valve, t): np.array(previous_solution["valve_status"][valve][t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods if t <= up_to}
    terms = [(2 * value - 1) * model._pump_status[pump, t] for (pump, t), value in pump_incumbent.items()] + [(2 * value - 1) * model._valve_status[valve, t] for (valve, t), value in valve_incumbent.items()]
    if not terms: return

    if not lazy_flag:
        model.addConstr(quicksum(terms) <= sum(pump_incumbent.values()) + sum(valve_incumbent.values()) - 1)
    else:
        model.cbLazy(quicksum(terms) <= sum(pump_incumbent.values()) + sum(valve_incumbent.values()) - 1)

def GurobiOperationalModelSoc(model_name, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, binary_pipe = False, binary_pump_valve = True, valid_inequalities = None):
    model_env = Env(empty=True)
    model_env.start()
    model = Model(model_name, env=model_env)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model, binary_pump_valve)
    DefineObjective(model)
    AddHeadLossConstraints(model, binary_pipe)
    AddPipeFacets(model)
    AddValveFlowConstraints(model)
    AddValveHeadConstraints(model)
    AddValveFacets(model)
    AddPumpHeadConstraints(model)
    AddPumpFacets(model)
    AddTankBalanceConstraints(model, hydraulic_timestep)
    AddJunctionBalanceConstraints(model)
    if valid_inequalities: AddValidInequalities(model, valid_inequalities)

    #lp_filename=os.path.join(results_directory, f'{model_name}.lp')
    #model.write(lp_filename)
    #model.setParam('MIPFocus', 3)
    model.setParam('Threads', 12)
    log_file_path = os.path.join(results_directory, f'{model_name}_log.txt')
    model.setParam('LogFile', log_file_path)
    model.setParam("NonConvex", 0)

    return model, model_env

def SolveGurobiOperationalModelSoc(model, model_name, results_directory, compute_iss=False):
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
            solution['pump_status'] = {pump: np.rint([model._pump_status[pump, t].x for t in model._set_of_time_periods]) for pump in model._set_of_pumps}

        if len(model._set_of_valves) > 0:
            solution['valve_status'] = {valve: np.rint([model._valve_status[valve, t].x for t in model._set_of_time_periods]) for valve in model._set_of_valves}

    return problem_status, solution, lower_bound
