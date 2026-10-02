import os
import numpy as np
from mosek.fusion import *
import mosek.fusion.pythonic

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_time_periods = list(range(0, model._time_steps))
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_links = list(model._links.keys())
    model._node_index = {label: index for index, label in enumerate(model._set_of_nodes)}
    model._link_index = {label: index for index, label in enumerate(model._set_of_links)}

def DefineDecisionVariables(model, binary_pump_valve):
    if model._set_of_nodes:
        model._head = model.variable('head', [len(model._set_of_nodes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([model._nodes[n].head_lower_bound[:model._time_steps] for n in model._set_of_nodes]), np.stack([model._nodes[n].head_upper_bound[:model._time_steps] for n in model._set_of_nodes])))

    if model._set_of_tanks:
        model._tank_terminal_head = model.variable('tank_terminal_head', [len(model._set_of_tanks)], Domain.inRange(
            np.array([model._tanks[t].head_lower_bound[model._time_steps] for t in model._set_of_tanks]), np.array([model._tanks[t].head_upper_bound[model._time_steps] for t in model._set_of_tanks])))

    if model._set_of_links:
        model._flow = model.variable('flow', [len(model._set_of_links), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([model._links[l].flow_lower_bound for l in model._set_of_links]), np.stack([model._links[l].flow_upper_bound for l in model._set_of_links])))

    if model._set_of_pumps:
        if binary_pump_valve:
            model._pump_status = model.variable('pump_status', [len(model._set_of_pumps), len(model._set_of_time_periods)], Domain.binary())
        else:
            model._pump_status = model.variable('pump_status', [len(model._set_of_pumps), len(model._set_of_time_periods)], Domain.inRange(0.0, 1.0))

    if model._set_of_valves:
        if binary_pump_valve:
            model._valve_status = model.variable('valve_status', [len(model._set_of_valves), len(model._set_of_time_periods)], Domain.binary())
        else:
            model._valve_status = model.variable('valve_status', [len(model._set_of_valves), len(model._set_of_time_periods)], Domain.inRange(0.0, 1.0))

def DefineObjective(model):
    if len(model._set_of_pumps) > 0:
        model.objective('objective', ObjectiveSense.Minimize, Expr.dot(np.stack([pump.operational_cost for pump in model._pumps.values()]), Expr.vstack([model._flow[model._link_index[pump.label], :] for pump in model._pumps.values()])) + Expr.dot(np.stack([pump.fixed_cost for pump in model._pumps.values()]), model._pump_status))
    else:
        model.objective('objective', ObjectiveSense.Minimize, Expr.constTerm(0.0))

def DefineBoundParameters(model):
    if len(model._set_of_nodes) > 0:
        model._head_lb = model.parameter('head_lb', [len(model._set_of_nodes), len(model._set_of_time_periods)])
        model._head_ub = model.parameter('head_ub', [len(model._set_of_nodes), len(model._set_of_time_periods)])
        model._head_lb.setValue(np.stack([model._nodes[n].head_lower_bound[:model._time_steps] for n in model._set_of_nodes]))
        model._head_ub.setValue(np.stack([model._nodes[n].head_upper_bound[:model._time_steps] for n in model._set_of_nodes]))

    if len(model._set_of_links) > 0:
        model._flow_lb = model.parameter('flow_lb', [len(model._set_of_links), len(model._set_of_time_periods)])
        model._flow_ub = model.parameter('flow_ub', [len(model._set_of_links), len(model._set_of_time_periods)])
        model._flow_lb.setValue(np.stack([model._links[l].flow_lower_bound for l in model._set_of_links]))
        model._flow_ub.setValue(np.stack([model._links[l].flow_upper_bound for l in model._set_of_links]))

        model._flow_lb_pos = model.parameter('flow_lb_pos', [len(model._set_of_links), len(model._set_of_time_periods)])
        model._flow_ub_neg = model.parameter('flow_ub_neg', [len(model._set_of_links), len(model._set_of_time_periods)])
        model._flow_lb_pos.setValue(np.stack([np.maximum(model._links[l].flow_lower_bound, 0.0) for l in model._set_of_links]))
        model._flow_ub_neg.setValue(np.stack([np.minimum(model._links[l].flow_upper_bound, 0.0) for l in model._set_of_links]))

def AddPipeFacets(model):
    model._pipe_facet_cons = []
    for i, pipe in enumerate(model._pipes.values()):
        for (t, orthant), facets in pipe.hull_facets.items():
            for face in facets:
                a, b, c, d, sense = face
                if sense == 'ge':
                    model._pipe_facet_cons.append(model.constraint(a * (model._head_pipe_start_plus[i, t] if orthant == 'pos' else model._head_pipe_start_minus[i, t]) + b * (model._head_pipe_end_plus[i, t] if orthant == 'pos' else model._head_pipe_end_minus[i, t]) + c * (model._flow_pipe_plus[i, t] if orthant == 'pos' else model._flow_pipe_minus[i, t]) >= d * (model._flow_pipe_status[i, t] if orthant == 'pos' else 1 - model._flow_pipe_status[i, t])))
                elif sense == 'le':
                    model._pipe_facet_cons.append(model.constraint(a * (model._head_pipe_start_plus[i, t] if orthant == 'pos' else model._head_pipe_start_minus[i, t]) + b * (model._head_pipe_end_plus[i, t] if orthant == 'pos' else model._head_pipe_end_minus[i, t]) + c * (model._flow_pipe_plus[i, t] if orthant == 'pos' else model._flow_pipe_minus[i, t]) <= d * (model._flow_pipe_status[i, t] if orthant == 'pos' else 1 - model._flow_pipe_status[i, t])))
                else:
                    model._pipe_facet_cons.append(model.constraint(a * (model._head_pipe_start_plus[i, t] if orthant == 'pos' else model._head_pipe_start_minus[i, t]) + b * (model._head_pipe_end_plus[i, t] if orthant == 'pos' else model._head_pipe_end_minus[i, t]) + c * (model._flow_pipe_plus[i, t] if orthant == 'pos' else model._flow_pipe_minus[i, t]) == d * (model._flow_pipe_status[i, t] if orthant == 'pos' else 1 - model._flow_pipe_status[i, t])))

def AddValveFacets(model):
    model._valve_facet_cons = []
    for i, valve in enumerate(model._valves.values()):
        link_index = model._link_index[valve.label]
        for (t, orthant), facets in valve.hull_facets.items():
            for face in facets:
                a, b, c, d, sense = face
                if sense == 'ge':
                    model._valve_facet_cons.append(model.constraint(a * model._head_valve_start_plus[i, t] + b * model._head_valve_end_plus[i, t] + c * model._flow[link_index, t] >= d * model._valve_status[i, t]))
                elif sense == 'le':
                    model._valve_facet_cons.append(model.constraint(a * model._head_valve_start_plus[i, t] + b * model._head_valve_end_plus[i, t] + c * model._flow[link_index, t] <= d * model._valve_status[i, t]))
                else:
                    model._valve_facet_cons.append(model.constraint(a * model._head_valve_start_plus[i, t] + b * model._head_valve_end_plus[i, t] + c * model._flow[link_index, t] == d * model._valve_status[i, t]))

def AddPumpFacets(model):
    model._pump_facet_cons = []
    for i, pump in enumerate(model._pumps.values()):
        link_index = model._link_index[pump.label]
        for t, facets in pump.hull_facets.items():
            for face in facets:
                a, b, c, d, sense = face
                if sense == 'ge':
                    model._pump_facet_cons.append(model.constraint(a * model._head_pump_start_one[i, t] + b * model._head_pump_end_one[i, t] + c * model._flow[link_index, t] >= d * model._pump_status[i, t]))
                elif sense == 'le':
                    model._pump_facet_cons.append(model.constraint(a * model._head_pump_start_one[i, t] + b * model._head_pump_end_one[i, t] + c * model._flow[link_index, t] <= d * model._pump_status[i, t]))
                else:
                    model._pump_facet_cons.append(model.constraint(a * model._head_pump_start_one[i, t] + b * model._head_pump_end_one[i, t] + c * model._flow[link_index, t] == d * model._pump_status[i, t]))

def AddHeadLossConstraints(model, binary_pipe): # May want to adjust coefficients for ill-posed models

    if len(model._set_of_pipes) > 0:
        model._flow_pipe_plus = model.variable('flow_pipe_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(
            np.zeros((len(model._set_of_pipes), len(model._set_of_time_periods))), np.stack([np.maximum(model._pipes[p].flow_upper_bound, 0.0) for p in model._set_of_pipes])))

        model._flow_pipe_minus = model.variable('flow_pipe_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pipes[p].flow_lower_bound, 0.0) for p in model._set_of_pipes]), np.zeros((len(model._set_of_pipes), len(model._set_of_time_periods)))))

        model._head_pipe_start_plus = model.variable('head_pipe_start_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pipes[p].start_node.head_lower_bound[:model._time_steps], 0.0) for p in model._set_of_pipes]), np.stack([np.maximum(model._pipes[p].start_node.head_upper_bound[:model._time_steps], 0.0) for p in model._set_of_pipes])))

        model._head_pipe_start_minus = model.variable('head_pipe_start_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pipes[p].start_node.head_lower_bound[:model._time_steps], 0.0) for p in model._set_of_pipes]), np.stack([np.maximum(model._pipes[p].start_node.head_upper_bound[:model._time_steps], 0.0) for p in model._set_of_pipes])))

        model._head_pipe_end_plus = model.variable('head_pipe_end_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pipes[p].end_node.head_lower_bound[:model._time_steps], 0.0) for p in model._set_of_pipes]), np.stack([np.maximum(model._pipes[p].end_node.head_upper_bound[:model._time_steps], 0.0) for p in model._set_of_pipes])))

        model._head_pipe_end_minus = model.variable('head_pipe_end_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pipes[p].end_node.head_lower_bound[:model._time_steps], 0.0) for p in model._set_of_pipes]), np.stack([np.maximum(model._pipes[p].end_node.head_upper_bound[:model._time_steps], 0.0) for p in model._set_of_pipes])))

        if binary_pipe:
            model._flow_pipe_status = model.variable('flow_pipe_status', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.binary())
        else:
            model._flow_pipe_status = model.variable('flow_pipe_status', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.inRange(0.0, 1.0))

    for i, pipe in enumerate(model._pipes.values()):
        link_index = model._link_index[pipe.label]
        for t in model._set_of_time_periods:
            if pipe.flow_upper_bound[t] == pipe.flow_lower_bound[t]:
                model.constraint(f'cons_head_loss_eq_{pipe.label}_{t}', model._head[model._node_index[pipe.start_node.label], t] - model._head[model._node_index[pipe.end_node.label], t] == pipe.loss_coefficient * (1 if pipe.flow_upper_bound[t] > 0.0 else -1) * (abs(pipe.flow_upper_bound[t]) ** pipe.exponent))
                model.constraint(f'fix_status_{pipe.label}_{t}', model._flow_pipe_status[i, t], Domain.equalsTo(1))
                model.constraint(f'fix_head_plus_start_{pipe.label}_{t}', model._head_pipe_start_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_plus_end_{pipe.label}_{t}', model._head_pipe_end_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_plus_flow_{pipe.label}_{t}', model._flow_pipe_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_minus_flow_{pipe.label}_{t}', model._flow_pipe_minus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_minus_start_{pipe.label}_{t}', model._head_pipe_start_minus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_minus_end_{pipe.label}_{t}', model._head_pipe_end_minus[i, t], Domain.equalsTo(0.0))
                continue

            model.constraint(f'cons_flow_{pipe.label}_{t}', model._flow[link_index, t] == model._flow_pipe_plus[i, t] + model._flow_pipe_minus[i, t])
            model.constraint(f'cons_head_start_{pipe.label}_{t}', model._head[model._node_index[pipe.start_node.label], t] == model._head_pipe_start_plus[i, t] + model._head_pipe_start_minus[i, t])
            model.constraint(f'cons_head_end_{pipe.label}_{t}', model._head[model._node_index[pipe.end_node.label], t] == model._head_pipe_end_plus[i, t] + model._head_pipe_end_minus[i, t])

            if pipe.flow_upper_bound[t] > 0.0:
                model.constraint(f'cons_head_start_plus_l_{pipe.label}_{t}', model._head_pipe_start_plus[i, t]  >= model._flow_pipe_status[i, t] * model._head_lb.pick([[model._node_index[pipe.start_node.label], t]]))
                model.constraint(f'cons_head_start_plus_u_{pipe.label}_{t}', model._head_pipe_start_plus[i, t]  <= model._flow_pipe_status[i, t] * model._head_ub.pick([[model._node_index[pipe.start_node.label], t]]))
                model.constraint(f'cons_head_end_plus_l_{pipe.label}_{t}', model._head_pipe_end_plus[i, t]  >= model._flow_pipe_status[i, t] * model._head_lb.pick([[model._node_index[pipe.end_node.label], t]]))
                model.constraint(f'cons_head_end_plus_u_{pipe.label}_{t}', model._head_pipe_end_plus[i, t]  <= model._flow_pipe_status[i, t] * model._head_ub.pick([[model._node_index[pipe.end_node.label], t]]))
                model.constraint(f'cons_plus_flow_l_{pipe.label}_{t}', model._flow_pipe_plus[i, t] >= model._flow_pipe_status[i, t] * model._flow_lb_pos.pick([[link_index, t]]))
                model.constraint(f'cons_plus_flow_u_{pipe.label}_{t}', model._flow_pipe_plus[i, t] <= model._flow_pipe_status[i, t] * model._flow_ub.pick([[link_index, t]]))
                #model.constraint(Expr.vstack(model._head_pipe_start_plus[i, t] - model._head_pipe_end_plus[i, t], (pipe.loss_coefficient ** (1/(1-pipe.exponent))) * model._flow_pipe_status[i, t], model._flow_pipe_plus[i, t]), Domain.inPPowerCone(1/pipe.exponent))
                model.constraint(Expr.vstack(model._head_pipe_start_plus[i, t] - model._head_pipe_end_plus[i, t], model._flow_pipe_status[i, t], (pipe.loss_coefficient ** (1 / pipe.exponent)) * model._flow_pipe_plus[i, t]), Domain.inPPowerCone(1 / pipe.exponent))
                #if pipe.flow_upper_bound[t] != max(0.0, pipe.flow_lower_bound[t]):
                    #model.constraint(f'cons_secant_plane_plus_{pipe.label}_{t}', model._head_pipe_start_plus[i, t] - model._head_pipe_end_plus[i, t] <= (pipe.loss_coefficient * (((model._flow_ub.pick([[link_index, t]]) ** pipe.exponent) - (max(0.0, model._flow_lb.pick([[link_index, t]])) ** pipe.exponent)) / (model._flow_ub.pick([[link_index, t]]) - max(0.0, model._flow_lb.pick([[link_index, t]])))) * (model._flow_pipe_plus[i, t] - (max(0.0, model._flow_lb.pick([[link_index, t]])) * model._flow_pipe_status[i, t]))) + (pipe.loss_coefficient * (max(0.0, model._flow_lb.pick([[link_index, t]])) ** pipe.exponent) * model._flow_pipe_status[i, t]))

            else:
                model.constraint(f'fix_head_plus_start_{pipe.label}_{t}', model._head_pipe_start_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_plus_end_{pipe.label}_{t}', model._head_pipe_end_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_status_off_{pipe.label}_{t}', model._flow_pipe_status[i, t], Domain.equalsTo(0))

            if pipe.flow_lower_bound[t] < 0.0:
                model.constraint(f'cons_head_start_minus_l_{pipe.label}_{t}', model._head_pipe_start_minus[i, t]  >= (1 - model._flow_pipe_status[i, t]) * model._head_lb.pick([[model._node_index[pipe.start_node.label], t]]))
                model.constraint(f'cons_head_start_minus_u_{pipe.label}_{t}', model._head_pipe_start_minus[i, t]  <= (1 - model._flow_pipe_status[i, t]) * model._head_ub.pick([[model._node_index[pipe.start_node.label], t]]))
                model.constraint(f'cons_head_end_minus_l_{pipe.label}_{t}', model._head_pipe_end_minus[i, t]  >= (1 - model._flow_pipe_status[i, t]) * model._head_lb.pick([[model._node_index[pipe.end_node.label], t]]))
                model.constraint(f'cons_head_end_minus_u_{pipe.label}_{t}', model._head_pipe_end_minus[i, t]  <= (1 - model._flow_pipe_status[i, t]) * model._head_ub.pick([[model._node_index[pipe.end_node.label], t]]))
                model.constraint(f'cons_minus_flow_l_{pipe.label}_{t}', model._flow_pipe_minus[i, t] >= (1 - model._flow_pipe_status[i, t]) * model._flow_lb.pick([[link_index, t]]))
                model.constraint(f'cons_minus_flow_u_{pipe.label}_{t}', model._flow_pipe_minus[i, t] <= (1 - model._flow_pipe_status[i, t]) * model._flow_ub_neg.pick([[link_index, t]]))
                #model.constraint(Expr.vstack(model._head_pipe_end_minus[i, t] - model._head_pipe_start_minus[i, t], (pipe.loss_coefficient ** (1/(1-pipe.exponent))) * (1 - model._flow_pipe_status[i, t]), model._flow_pipe_minus[i, t]), Domain.inPPowerCone(1/pipe.exponent))
                model.constraint(Expr.vstack(model._head_pipe_end_minus[i, t] - model._head_pipe_start_minus[i, t], 1 - model._flow_pipe_status[i, t], (pipe.loss_coefficient ** (1 / pipe.exponent)) * model._flow_pipe_minus[i, t]), Domain.inPPowerCone(1/pipe.exponent))
                #if pipe.flow_lower_bound[t] != min(0.0, pipe.flow_upper_bound[t]):
                    #model.constraint(f'cons_secant_plane_minus_{pipe.label}_{t}', model._head_pipe_start_minus[i, t] - model._head_pipe_end_minus[i, t] >= (pipe.loss_coefficient * ((((-model._flow_lb.pick([[link_index, t]])) ** pipe.exponent) - ((-min(0.0, model._flow_ub.pick([[link_index, t]]))) ** pipe.exponent)) / (min(0.0, model._flow_ub.pick([[link_index, t]])) - model._flow_lb.pick([[link_index, t]]))) * (model._flow_pipe_minus[i, t] - (model._flow_lb.pick([[link_index, t]]) * (1 - model._flow_pipe_status[i, t])))) - (pipe.loss_coefficient * ((-model._flow_lb.pick([[link_index, t]])) ** pipe.exponent) * (1 - model._flow_pipe_status[i, t])))

            elif pipe.flow_upper_bound[t] > 0.0:
                model.constraint(f'fix_head_minus_start_{pipe.label}_{t}', model._head_pipe_start_minus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_minus_end_{pipe.label}_{t}', model._head_pipe_end_minus[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_status_on_{pipe.label}_{t}', model._flow_pipe_status[i, t], Domain.equalsTo(1))

def AddValveFlowConstraints(model):
    for i, valve in enumerate(model._valves.values()):
        link_index = model._link_index[valve.label]
        for t in model._set_of_time_periods:
            if valve.decisions[t] == 1:
                model.constraint(f'fix_status_on_{valve.label}_{t}', model._valve_status[i, t], Domain.equalsTo(1))
            elif valve.decisions[t] == 0:
                model.constraint(f'fix_status_off_{valve.label}_{t}', model._valve_status[i, t], Domain.equalsTo(0))
                model.constraint(f'fix_flow_{valve.label}_{t}', model._flow[link_index, t], Domain.equalsTo(0.0))
            else:
                model.constraint(f'cons_valve_flow_u_{valve.label}_{t}', (model._valve_status[i, t] * model._flow_ub.pick([[link_index, t]])) - model._flow[link_index, t] >= 0)
                model.constraint(f'cons_valve_flow_l_{valve.label}_{t}', model._flow[link_index, t] - (model._valve_status[i, t] * model._flow_lb.pick([[link_index, t]])) >= 0)

def AddPumpHeadConstraints(model): # Will focus on coefficient adjustments for ill posed models later
    if len(model._set_of_pumps) > 0:
        model._head_pump_start_one = model.variable('head_pump_start_one', [len(model._set_of_pumps), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pumps[p].start_node.head_lower_bound[:model._time_steps], 0.0) for p in model._set_of_pumps]), np.stack([np.maximum(model._pumps[p].start_node.head_upper_bound[:model._time_steps], 0.0) for p in model._set_of_pumps])))

        model._head_pump_end_one = model.variable('head_pump_end_one', [len(model._set_of_pumps), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([np.minimum(model._pumps[p].end_node.head_lower_bound[:model._time_steps], 0.0) for p in model._set_of_pumps]), np.stack([np.maximum(model._pumps[p].end_node.head_upper_bound[:model._time_steps], 0.0) for p in model._set_of_pumps])))

    for i, pump in enumerate(model._pumps.values()):
        link_index = model._link_index[pump.label]
        for t in model._set_of_time_periods:
            if pump.decisions[t] == 0:
                model.constraint(f'fix_flow_{pump.label}_{t}', model._flow[link_index, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_status_off_{pump.label}_{t}', model._pump_status[i, t], Domain.equalsTo(0))
                model.constraint(f'fix_head_start_one_{pump.label}_{t}', model._head_pump_start_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_pump_end_one_{pump.label}_{t}', model._head_pump_end_one[i, t], Domain.equalsTo(0.0))
                continue

            if pump.flow_lower_bound[t] == pump.flow_upper_bound[t] and pump.flow_upper_bound[t] > 0.0:
                model.constraint(f'fix_status_on_{pump.label}_{t}', model._pump_status[i, t], Domain.equalsTo(1))
                model.constraint(f'fix_head_start_one_{pump.label}_{t}', model._head_pump_start_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_pump_end_one_{pump.label}_{t}', model._head_pump_end_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'cons_head_gain_{pump.label}_{t}', model._head[model._node_index[pump.start_node.label], t] - model._head[model._node_index[pump.end_node.label], t] + pump.curve_a + (pump.curve_b * (pump.flow_upper_bound[t] ** pump.curve_c)) == 0)
                continue

            if pump.decisions[t] == 1:
                model.constraint(f'fix_status_on_{pump.label}_{t}', model._pump_status[i, t], Domain.equalsTo(1))

            if pump.flow_lower_bound[t] == pump.flow_upper_bound[t] == 0.0:
                model.constraint(f'fix_head_start_one_{pump.label}_{t}', model._head_pump_start_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_pump_end_one_{pump.label}_{t}', model._head_pump_end_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'cons_head_gain_l_{pump.label}_{t}', model._head[model._node_index[pump.start_node.label], t] - model._head[model._node_index[pump.end_node.label], t] + pump.curve_a * model._pump_status[i, t] >= model._head_lb.pick([[model._node_index[pump.start_node.label], t]]) - model._head_ub.pick([[model._node_index[pump.end_node.label], t]]) - model._head_lb.pick([[model._node_index[pump.start_node.label], t]]) * model._pump_status[i, t] + model._head_ub.pick([[model._node_index[pump.end_node.label], t]]) * model._pump_status[i, t])
                model.constraint(f'cons_head_gain_u_{pump.label}_{t}', model._head[model._node_index[pump.start_node.label], t] - model._head[model._node_index[pump.end_node.label], t] + pump.curve_a * model._pump_status[i, t] <= model._head_ub.pick([[model._node_index[pump.start_node.label], t]]) - model._head_lb.pick([[model._node_index[pump.end_node.label], t]]) - model._head_ub.pick([[model._node_index[pump.start_node.label], t]]) * model._pump_status[i, t] + model._head_lb.pick([[model._node_index[pump.end_node.label], t]]) * model._pump_status[i, t])
                continue

            if pump.curve_c == 1:
                model.constraint(f'fix_head_start_one_{pump.label}_{t}', model._head_pump_start_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'fix_head_pump_end_one_{pump.label}_{t}', model._head_pump_end_one[i, t], Domain.equalsTo(0.0))
                model.constraint(f'cons_head_gain_l_{pump.label}_{t}', model._head_lb.pick([[model._node_index[pump.start_node.label], t]]) - model._head_ub.pick([[model._node_index[pump.end_node.label], t]]) - model._head_lb.pick([[model._node_index[pump.start_node.label], t]]) * model._pump_status[i, t] + model._head_ub.pick([[model._node_index[pump.end_node.label], t]]) * model._pump_status[i, t] <= model._head[model._node_index[pump.start_node.label], t] - model._head[model._node_index[pump.end_node.label], t] + pump.curve_a * model._pump_status[i, t] + pump.curve_b * model._flow[link_index, t])
                model.constraint(f'cons_head_gain_u_{pump.label}_{t}', model._head[model._node_index[pump.start_node.label], t] - model._head[model._node_index[pump.end_node.label], t] + pump.curve_a * model._pump_status[i, t] + pump.curve_b * model._flow[link_index, t] <= model._head_ub.pick([[model._node_index[pump.start_node.label], t]]) - model._head_lb.pick([[model._node_index[pump.end_node.label], t]]) - model._head_ub.pick([[model._node_index[pump.start_node.label], t]]) * model._pump_status[i, t] + model._head_lb.pick([[model._node_index[pump.end_node.label], t]]) * model._pump_status[i, t])
                model.constraint(f'cons_flow_pump_l_{pump.label}_{t}', model._flow[link_index, t] >= model._pump_status[i, t] * model._flow_lb.pick([[link_index, t]]))
                model.constraint(f'cons_flow_pump_u_{pump.label}_{t}', model._flow[link_index, t] <= model._pump_status[i, t] * model._flow_ub.pick([[link_index, t]]))
                continue

            model.constraint(f'cons_head_pump_start_one_l_{pump.label}_{t}', model._head_pump_start_one[i, t] >= model._pump_status[i, t] * model._head_lb.pick([[model._node_index[pump.start_node.label], t]]))
            model.constraint(f'cons_head_pump_start_one_u_{pump.label}_{t}', model._head_pump_start_one[i, t] <= model._pump_status[i, t] * model._head_ub.pick([[model._node_index[pump.start_node.label], t]]))
            model.constraint(f'cons_head_pump_end_one_l_{pump.label}_{t}', model._head_pump_end_one[i, t] >= model._pump_status[i, t] * model._head_lb.pick([[model._node_index[pump.end_node.label], t]]))
            model.constraint(f'cons_head_pump_end_one_u_{pump.label}_{t}', model._head_pump_end_one[i, t] <= model._pump_status[i, t] * model._head_ub.pick([[model._node_index[pump.end_node.label], t]]))
            model.constraint(f'cons_head_pump_start_zero_l_{pump.label}_{t}', model._head[model._node_index[pump.start_node.label], t] - model._head_pump_start_one[i, t] >= (1 - model._pump_status[i, t]) * model._head_lb.pick([[model._node_index[pump.start_node.label], t]]))
            model.constraint(f'cons_head_pump_start_zero_u_{pump.label}_{t}', model._head[model._node_index[pump.start_node.label], t] - model._head_pump_start_one[i, t] <= (1 - model._pump_status[i, t]) * model._head_ub.pick([[model._node_index[pump.start_node.label], t]]))
            model.constraint(f'cons_head_pump_end_zero_l_{pump.label}_{t}', model._head[model._node_index[pump.end_node.label], t] - model._head_pump_end_one[i, t] >= (1 - model._pump_status[i, t]) * model._head_lb.pick([[model._node_index[pump.end_node.label], t]]))
            model.constraint(f'cons_head_pump_end_zero_u_{pump.label}_{t}', model._head[model._node_index[pump.end_node.label], t] - model._head_pump_end_one[i, t] <= (1 - model._pump_status[i, t]) * model._head_ub.pick([[model._node_index[pump.end_node.label], t]]))
            model.constraint(f'cons_flow_pump_one_l_{pump.label}_{t}', model._flow[link_index, t] >= model._pump_status[i, t] * model._flow_lb.pick([[link_index, t]]))
            model.constraint(f'cons_flow_pump_one_u_{pump.label}_{t}', model._flow[link_index, t] <= model._pump_status[i, t] * model._flow_ub.pick([[link_index, t]]))
            #model.constraint(Expr.vstack(model._head_pump_start_one[i, t] - model._head_pump_end_one[i, t] + pump.curve_a * model._pump_status[i, t], ((-pump.curve_b) ** (1 / (1 - pump.curve_c))) * model._pump_status[i, t], model._flow[link_index, t]), Domain.inPPowerCone(1/pump.curve_c))
            model.constraint(Expr.vstack(model._head_pump_start_one[i, t] - model._head_pump_end_one[i, t] + pump.curve_a * model._pump_status[i, t], model._pump_status[i, t], ((-pump.curve_b) ** (1 / pump.curve_c)) * model._flow[link_index, t]), Domain.inPPowerCone(1/pump.curve_c))
            #if pump.flow_lower_bound[t] != pump.flow_upper_bound[t]:
                #model.constraint(f'cons_secant_plane_{pump.label}_{t}', model._head_gain_flow[i, t] <= model._pump_status[i, t] * (model._flow_lb.pick([[link_index, t]]) ** pump.curve_c) + (((model._flow_ub.pick([[link_index, t]]) ** pump.curve_c) - (model._flow_lb.pick([[link_index, t]]) ** pump.curve_c)) / (model._flow_ub.pick([[link_index, t]]) - model._flow_lb.pick([[link_index, t]]))) * (model._flow[link_index, t] - (model._flow_lb.pick([[link_index, t]]) * model._pump_status[i, t])))

def AddValveHeadConstraints(model):
    model._head_valve_start_plus = model.variable('head_valve_start_plus', [len(model._set_of_valves), len(model._set_of_time_periods)], Domain.inRange(np.stack([np.minimum(valve.start_node.head_lower_bound[:model._time_steps], 0.0) for valve in model._valves.values()]), np.stack([np.maximum(valve.start_node.head_upper_bound[:model._time_steps], 0.0) for valve in model._valves.values()])))
    model._head_valve_end_plus = model.variable('head_valve_end_plus', [len(model._set_of_valves), len(model._set_of_time_periods)], Domain.inRange(np.stack([np.minimum(valve.end_node.head_lower_bound[:model._time_steps], 0.0) for valve in model._valves.values()]), np.stack([np.maximum(valve.end_node.head_upper_bound[:model._time_steps], 0.0) for valve in model._valves.values()])))

    for i, valve in enumerate(model._valves.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_valve_head_start_zero_l_{valve.label}_{t}', (model._head[model._node_index[valve.start_node.label], t] - model._head_valve_start_plus[i, t]) >= (1 - model._valve_status[i, t]) * model._head_lb.pick([[model._node_index[valve.start_node.label], t]]))
            model.constraint(f'cons_valve_head_start_zero_u_{valve.label}_{t}', (model._head[model._node_index[valve.start_node.label], t] - model._head_valve_start_plus[i, t]) <= (1 - model._valve_status[i, t]) * model._head_ub.pick([[model._node_index[valve.start_node.label], t]]))
            model.constraint(f'cons_valve_head_end_zero_l_{valve.label}_{t}', (model._head[model._node_index[valve.end_node.label], t] - model._head_valve_end_plus[i, t]) >= (1 - model._valve_status[i, t]) * model._head_lb.pick([[model._node_index[valve.end_node.label], t]]))
            model.constraint(f'cons_valve_head_end_zero_u_{valve.label}_{t}', (model._head[model._node_index[valve.end_node.label], t] - model._head_valve_end_plus[i, t]) <= (1 - model._valve_status[i, t]) * model._head_ub.pick([[model._node_index[valve.end_node.label], t]]))

            model.constraint(f'cons_valve_head_start_plus_l_{valve.label}_{t}', model._head_valve_start_plus[i, t] >= model._valve_status[i, t] * model._head_lb.pick([[model._node_index[valve.start_node.label], t]]))
            model.constraint(f'cons_valve_head_start_plus_u_{valve.label}_{t}', model._head_valve_start_plus[i, t] <= model._valve_status[i, t] * model._head_ub.pick([[model._node_index[valve.start_node.label], t]]))
            model.constraint(f'cons_valve_head_end_plus_l_{valve.label}_{t}', model._head_valve_end_plus[i, t] >= model._valve_status[i, t] * model._head_lb.pick([[model._node_index[valve.end_node.label], t]]))
            model.constraint(f'cons_valve_head_end_plus_u_{valve.label}_{t}', model._head_valve_end_plus[i, t] <= model._valve_status[i, t] * model._head_ub.pick([[model._node_index[valve.end_node.label], t]]))

            if valve.flow_lower_bound[t] == valve.flow_upper_bound[t]:
                model.constraint(f'cons_valve_fixed_head_loss_{valve.label}_{t}', model._head_valve_start_plus[i, t] - model._head_valve_end_plus[i, t] == valve.loss_coefficient * (valve.flow_lower_bound[t] ** valve.exponent) * model._valve_status[i, t])
            elif valve.decisions[t] != 0:
                model.constraint(Expr.vstack(model._head_valve_start_plus[i, t] - model._head_valve_end_plus[i, t], model._valve_status[i, t], (valve.loss_coefficient ** (1 / valve.exponent)) * model._flow[model._link_index[valve.label], t]), Domain.inPPowerCone(1 / valve.exponent))

def AddTankBalanceConstraints(model, hydraulic_timestep):
    for i, tank in enumerate(model._tanks.values()):
        node_index = model._node_index[tank.label]
        incidence = np.zeros(len(model._set_of_links))
        for edge in tank.incoming_edges:
            incidence[model._link_index[edge.label]] += 1.0
        for edge in tank.outgoing_edges:
            incidence[model._link_index[edge.label]] -= 1.0
        for t in model._set_of_time_periods:
            if t != model._time_steps - 1:
                model.constraint(f'cons_tank_balance_{tank.label}_{t}', (hydraulic_timestep / tank.area) * Expr.dot(incidence, model._flow[:, t]) - model._head[node_index, t+1] + model._head[node_index, t] == 0)
            else:
                model.constraint(f'cons_tank_balance_{tank.label}_{t}', (hydraulic_timestep / tank.area) * Expr.dot(incidence, model._flow[:, t]) - model._tank_terminal_head[i] + model._head[node_index, t] == 0)

def AddJunctionBalanceConstraints(model):
    for junction in model._junctions.values():
        if junction.segment == 'reservoir':
            continue
        incidence = np.zeros(len(model._set_of_links))
        for edge in junction.incoming_edges:
            incidence[model._link_index[edge.label]] += 1.0
        for edge in junction.outgoing_edges:
            incidence[model._link_index[edge.label]] -= 1.0
        for t in model._set_of_time_periods:
            model.constraint(f'cons_junction_balance_{junction.label}_{t}', Expr.dot(incidence, model._flow[:, t]) - junction.net_demand[t] == 0)

def AddNoGoodCut(model, previous_solution, up_to=None):
    up_to = model._time_steps if up_to is None else up_to
    expression = []
    pump_incumbent = np.array([previous_solution['pump_status'][pump][:up_to] for pump in model._set_of_pumps], dtype=int).reshape(-1)
    valve_incumbent = np.array([previous_solution['valve_status'][valve][:up_to] for valve in model._set_of_valves], dtype=int).reshape(-1)

    if len(pump_incumbent) > 0:
        pump_coefficients = (2 * pump_incumbent - 1).astype(float)
        expression.append(Expr.dot(pump_coefficients.tolist(), Var.flatten(model._pump_status[:, :up_to])))
    if len(valve_incumbent) > 0:
        valve_coefficients = (2 * valve_incumbent - 1).astype(float)
        expression.append(Expr.dot(valve_coefficients.tolist(), Var.flatten(model._valve_status[:, :up_to])))

    model.constraint(Expr.add(expression) <= np.sum(pump_incumbent) + np.sum(valve_incumbent) - 1)

def MosekOperationalModel(model_name, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit, binary_pipe = False, binary_pump_valve = True, obbt = False):
    model = Model(model_name)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model, binary_pump_valve)
    DefineBoundParameters(model)

    if not obbt:
        DefineObjective(model)

    if len(model._set_of_pipes) > 0:
        AddHeadLossConstraints(model, binary_pipe)
        AddPipeFacets(model)

    if len(model._set_of_pumps) > 0:
        AddPumpHeadConstraints(model)
        AddPumpFacets(model)

    if len(model._set_of_valves) > 0:
        AddValveFlowConstraints(model)
        AddValveHeadConstraints(model)
        AddValveFacets(model)

    if len(model._set_of_tanks) > 0:
        AddTankBalanceConstraints(model, hydraulic_timestep)

    if len(model._set_of_junctions) > 0:
        AddJunctionBalanceConstraints(model)

    #model.setSolverParam('ptfWriteSolutions', 'on')
    model.setSolverParam('numThreads', 12)

    if (binary_pump_valve and (len(model._set_of_pumps) > 0 or len(model._set_of_valves) > 0)) or (binary_pipe and len(model._set_of_pipes) > 0):
        model.setSolverParam('mioMaxTime', time_limit)
    else:
        model.setSolverParam('optimizerMaxTime', time_limit)

    return model

def SolveMosekOperationalModel(model, model_name, results_directory, binary_pipe = False, binary_pump_valve = True):
    model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
    with open(os.path.join(results_directory, f'{model_name}_log.log'), 'w') as f:
        model.setLogHandler(f)
        model.solve()
    model.setLogHandler(None)
    #model.writeTask(os.path.join(results_directory, f'{model_name}_solution.ptf'))

    problem_status = 'feasible' if model.getPrimalSolutionStatus() in (SolutionStatus.Optimal, SolutionStatus.Feasible) else 'infeasible' if model.getProblemStatus() == ProblemStatus.PrimalInfeasible else 'nosolution'
    try:
        lower_bound = (model.getSolverDoubleInfo('mioObjBound') if (binary_pump_valve and (len(model._set_of_pumps) > 0 or len(model._set_of_valves) > 0)) or (binary_pipe and len(model._set_of_pipes) > 0) else model.dualObjValue()) if problem_status != 'infeasible' else None
    except SolutionError:
        lower_bound = None

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            head_levels = np.array(model._head.level()).reshape(len(model._set_of_nodes), len(model._set_of_time_periods))
            solution['head'] = {node: head_levels[i] for i, node in enumerate(model._set_of_nodes)}

        if len(model._set_of_tanks) > 0:
            terminal_head_levels = np.array(model._tank_terminal_head.level()).reshape(len(model._set_of_tanks))
            for i, tank in enumerate(model._set_of_tanks):
                solution['head'][tank] = np.append(solution['head'][tank], terminal_head_levels[i])

        if len(model._set_of_links) > 0:
            flow_solution_levels = np.array(model._flow.level()).reshape(len(model._set_of_links), len(model._set_of_time_periods))
            solution['flow'] = {link: flow_solution_levels[i] for i, link in enumerate(model._set_of_links)}

        if len(model._set_of_pumps) > 0:
            pump_status_solution_levels = np.rint(model._pump_status.level()).reshape(len(model._set_of_pumps), len(model._set_of_time_periods))
            solution['pump_status'] = {pump: pump_status_solution_levels[i] for i, pump in enumerate(model._set_of_pumps)}

        if len(model._set_of_valves) > 0:
            valve_status_solution_levels = np.rint(model._valve_status.level()).reshape(len(model._set_of_valves), len(model._set_of_time_periods))
            solution['valve_status'] = {valve: valve_status_solution_levels[i] for i, valve in enumerate(model._set_of_valves)}

    return problem_status, solution, lower_bound