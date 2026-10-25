import os
import numpy as np
from mosek.fusion import *
import mosek.fusion.pythonic

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_non_reservoir_junctions = [label for label, junction in model._junctions.items() if junction.segment != 'reservoir']
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_time_periods = list(range(0, model._time_steps))
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_arcs = list(model._arcs.keys())
    model._set_of_links = list(model._links.keys())
    model._set_of_pipe_families = list(model._pipe_families.keys())
    model._node_index = {label: index for index, label in enumerate(model._set_of_nodes)}
    model._tank_index = {label: index for index, label in enumerate(model._set_of_tanks)}
    model._arc_index = {label: index for index, label in enumerate(model._set_of_arcs)}
    model._link_index = {label: index for index, label in enumerate(model._set_of_links)}
    model._pipe_index = {label: i for i, label in enumerate(model._set_of_pipes)}
    model._pump_index = {label: i for i, label in enumerate(model._set_of_pumps)}
    model._valve_index = {label: i for i, label in enumerate(model._set_of_valves)}
    model._incoming_edges = {node: [] for node in model._set_of_nodes}
    model._outgoing_edges = {node: [] for node in model._set_of_nodes}

    for label, link in model._links.items():
        model._incoming_edges[link.end_node.label].append(label)
        model._outgoing_edges[link.start_node.label].append(label)

def DefineDecisionVariables(model):
    model._head = model.variable('head', Domain.inRange(model._head_lb.getValue(), model._head_ub.getValue(), model._head_lb.getShape()))
    model._flow = model.variable('flow', Domain.inRange(model._flow_lb.getValue(), model._flow_ub.getValue(), model._flow_lb.getShape()))

    if model._set_of_tanks:
        model._tank_terminal_head = model.variable('tank_terminal_head', Domain.inRange(model._tank_terminal_head_lb.getValue(), model._tank_terminal_head_ub.getValue()))

    if model._set_of_pumps:
        model._pump_status = model.variable('pump_status', Domain.inRange(model._pump_status_lb.getValue(), model._pump_status_ub.getValue(), model._pump_status_lb.getShape()))
        if model._binary_pump_valve: model._pump_status.makeInteger()

    if model._set_of_valves:
        model._valve_status = model.variable('valve_status', Domain.inRange(model._valve_status_lb.getValue(), model._valve_status_ub.getValue(), model._valve_status_lb.getShape()))
        if model._binary_pump_valve: model._valve_status.makeInteger()

    model._pipe_status = model.variable('pipe_status', Domain.inRange(model._pipe_status_lb.getValue(), model._pipe_status_ub.getValue())) if model._problem != 'operational' else {pipe: 1.0 for pipe in model._set_of_pipes}
    if model._problem != 'operational' and model._binary_pipe_selection: model._pipe_status.makeInteger()
    model._flow_status_pos = model.variable('flow_status_pos', Domain.inRange(0.0, model._flow_status_pos_ub.getValue(), model._flow_status_pos_ub.getShape()))
    model._flow_status_neg = model.variable('flow_status_neg', Domain.inRange(0.0, model._flow_status_neg_ub.getValue(), model._flow_status_neg_ub.getShape()))

    if model._binary_flow_direction:
        model._flow_status_pos.makeInteger()
        model._flow_status_neg.makeInteger()

    model._positive_indices = {(pipe,t): i for i, (pipe,t) in enumerate((pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods if model._pipes[pipe].selection_decision != 0 and model._pipes[pipe].direction_decision[t] != 0)}
    model._negative_indices = {(pipe,t): i for i, (pipe,t) in enumerate((pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods if model._pipes[pipe].selection_decision != 0 and model._pipes[pipe].direction_decision[t] != 1)}
    model._flow_pos_indices = {(pipe,t): i for i, (pipe,t) in enumerate((pipe,t) for pipe,t in model._positive_indices if not model._pipes[pipe].fixed_flow[t] and model._pipes[pipe].positive_flow_lower_bound[t] != model._pipes[pipe].flow_upper_bound[t])}
    model._flow_neg_indices = {(pipe,t): i for i, (pipe,t) in enumerate((pipe,t) for pipe,t in model._negative_indices if not model._pipes[pipe].fixed_flow[t] and model._pipes[pipe].flow_lower_bound[t] != model._pipes[pipe].negative_flow_upper_bound[t])}
    model._flow_pos = model.variable('flow_pos', [len(model._flow_pos_indices)], Domain.inRange(np.zeros(len(model._flow_pos_indices)), np.array([max(0.0, model._pipes[pipe].flow_upper_bound[t]) for pipe,t in model._flow_pos_indices])))
    model._flow_neg = model.variable('flow_neg', [len(model._flow_neg_indices)], Domain.inRange(np.array([min(0.0, model._pipes[pipe].flow_lower_bound[t]) for pipe,t in model._flow_neg_indices]), np.zeros(len(model._flow_neg_indices))))

    model._positive_indices.update({(valve,t): i for i, (valve,t) in enumerate(((valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods if model._valves[valve].status_decisions[t] != 0 and not model._valves[valve].fixed_flow[t]), start=len(model._positive_indices))})
    model._positive_indices.update({(pump,t): i for i, (pump,t) in enumerate(((pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods if model._pumps[pump].status_decisions[t] != 0 and not model._pumps[pump].fixed_flow[t] and model._pumps[pump].curve_c != 1.0), start=len(model._positive_indices))})
    model._head_start_pos = model.variable('head_start_pos', [len(model._positive_indices)], Domain.inRange(np.array([min(0.0, model._arcs[arc].start_node.head_lower_bound[t]) for arc,t in model._positive_indices]), np.array([max(0.0, model._arcs[arc].start_node.head_upper_bound[t]) for arc,t in model._positive_indices])))
    model._head_start_neg = model.variable('head_start_neg', [len(model._negative_indices)], Domain.inRange(np.array([min(0.0, model._pipes[pipe].start_node.head_lower_bound[t]) for pipe,t in model._negative_indices]), np.array([max(0.0, model._pipes[pipe].start_node.head_upper_bound[t]) for pipe,t in model._negative_indices])))
    model._head_end_pos = model.variable('head_end_pos', [len(model._positive_indices)], Domain.inRange(np.array([min(0.0, model._arcs[arc].end_node.head_lower_bound[t]) for arc,t in model._positive_indices]), np.array([max(0.0, model._arcs[arc].end_node.head_upper_bound[t]) for arc,t in model._positive_indices])))
    model._head_end_neg = model.variable('head_end_neg', [len(model._negative_indices)], Domain.inRange(np.array([min(0.0, model._pipes[pipe].end_node.head_lower_bound[t]) for pipe,t in model._negative_indices]), np.array([max(0.0, model._pipes[pipe].end_node.head_upper_bound[t]) for pipe,t in model._negative_indices])))

def DefineObjective(model):
    model.objective('objective', ObjectiveSense.Minimize, (Expr.dot([pump.fixed_cost[t] for pump in model._pumps.values() for t in model._set_of_time_periods], Var.flatten(model._pump_status)) + Expr.dot([pump.operational_cost[t] for pump in model._pumps.values() for t in model._set_of_time_periods], model._flow.pick([[model._link_index[pump.label],t] for pump in model._pumps.values() for t in model._set_of_time_periods])) if model._pumps else Expr.constTerm(0.0)) + (Expr.dot([pipe.installation_cost or 0.0 for pipe in model._pipes.values()], model._pipe_status) if model._problem != 'operational' else Expr.constTerm(0.0)))

def DefineBounds(model):
    bounds = {}
    bounds['head_lb'] = np.array([node.head_lower_bound[:model._time_steps] for node in model._nodes.values()])
    bounds['head_ub'] = np.array([node.head_upper_bound[:model._time_steps] for node in model._nodes.values()])
    bounds['flow_lb'] = np.array([[min((pipe.flow_lower_bound[t] for pipe in link.options if pipe.selection_decision != 0), default=0.0) if label in model._pipe_families else link.flow_lower_bound[t] for t in model._set_of_time_periods] for label, link in model._links.items()])
    bounds['flow_ub'] = np.array([[max((pipe.flow_upper_bound[t] for pipe in link.options if pipe.selection_decision != 0), default=0.0) if label in model._pipe_families else link.flow_upper_bound[t] for t in model._set_of_time_periods] for label, link in model._links.items()])
    bounds['arc_flow_lb'] = np.array([arc.flow_lower_bound for arc in model._arcs.values()])
    bounds['arc_flow_ub'] = np.array([arc.flow_upper_bound for arc in model._arcs.values()])
    bounds['arc_flow_lb_pos'] = np.array([arc.positive_flow_lower_bound for arc in model._arcs.values()])
    bounds['arc_flow_ub_neg'] = np.array([arc.negative_flow_upper_bound if label in model._pipes else np.zeros(model._time_steps) for label, arc in model._arcs.items()])
    bounds['flow_status_pos_ub'] = np.array([(pipe.direction_decision != 0) & (pipe.selection_decision != 0) for pipe in model._pipes.values()], dtype=float)
    bounds['flow_status_neg_ub'] = np.array([(pipe.direction_decision != 1) & (pipe.selection_decision != 0) for pipe in model._pipes.values()], dtype=float)

    if model._problem != 'operational':
        bounds['pipe_status_lb'] = np.array([pipe.selection_decision == 1 for pipe in model._pipes.values()], dtype=float)
        bounds['pipe_status_ub'] = np.array([pipe.selection_decision != 0 for pipe in model._pipes.values()], dtype=float)
    if model._set_of_tanks:
        bounds['tank_terminal_head_lb'] = np.array([tank.head_lower_bound[-1] for tank in model._tanks.values()])
        bounds['tank_terminal_head_ub'] = np.array([tank.head_upper_bound[-1] for tank in model._tanks.values()])
    if model._set_of_pumps:
        bounds['pump_status_lb'] = np.array([pump.status_decisions == 1 for pump in model._pumps.values()], dtype=float)
        bounds['pump_status_ub'] = np.array([pump.status_decisions != 0 for pump in model._pumps.values()], dtype=float)
    if model._set_of_valves:
        bounds['valve_status_lb'] = np.array([valve.status_decisions == 1 for valve in model._valves.values()], dtype=float)
        bounds['valve_status_ub'] = np.array([valve.status_decisions != 0 for valve in model._valves.values()], dtype=float)

    for name, values in bounds.items():
        parameter = model.parameter(name, list(values.shape))
        parameter.setValue(values)
        setattr(model, '_' + name, parameter)

def AddPipeFacets(model):
    constraints = []
    for i, pipe in enumerate(model._pipes.values()):
        for (t, orthant), facets in pipe.hull_facets.items():
            if (pipe.label,t) not in (model._flow_pos_indices if orthant == 'pos' else model._flow_neg_indices): continue
            index = (model._positive_indices if orthant == 'pos' else model._negative_indices)[pipe.label,t]
            for a, b, c, d, sense in facets:
                constraints.append(model.constraint(a * (model._head_start_pos[index] if orthant == 'pos' else model._head_start_neg[index]) + b * (model._head_end_pos[index] if orthant == 'pos' else model._head_end_neg[index]) + c * (model._flow_pos[model._flow_pos_indices[pipe.label,t]] if orthant == 'pos' else model._flow_neg[model._flow_neg_indices[pipe.label,t]]) - d * (model._flow_status_pos[i,t] if orthant == 'pos' else model._flow_status_neg[i,t]), Domain.greaterThan(0.0) if sense == 'ge' else Domain.lessThan(0.0) if sense == 'le' else Domain.equalsTo(0.0)))
    return constraints

def AddValveFacets(model):
    constraints = []
    for i, valve in enumerate(model._valves.values()):
        for (t, orthant), facets in valve.hull_facets.items():
            if (valve.label,t) not in model._positive_indices or valve.positive_flow_lower_bound[t] == valve.flow_upper_bound[t]: continue
            index = model._positive_indices[valve.label,t]
            for a, b, c, d, sense in facets:
                constraints.append(model.constraint(a * model._head_start_pos[index] + b * model._head_end_pos[index] + c * model._flow[model._link_index[valve.label],t] - d * model._valve_status[i,t], Domain.greaterThan(0.0) if sense == 'ge' else Domain.lessThan(0.0) if sense == 'le' else Domain.equalsTo(0.0)))
    return constraints

def AddPumpFacets(model):
    constraints = []
    for i, pump in enumerate(model._pumps.values()):
        for t, facets in pump.hull_facets.items():
            if (pump.label,t) not in model._positive_indices or pump.positive_flow_lower_bound[t] == pump.flow_upper_bound[t]: continue
            index = model._positive_indices[pump.label,t]
            for a, b, c, d, sense in facets:
                constraints.append(model.constraint(a * model._head_start_pos[index] + b * model._head_end_pos[index] + c * model._flow[model._link_index[pump.label],t] - d * model._pump_status[i,t], Domain.greaterThan(0.0) if sense == 'ge' else Domain.lessThan(0.0) if sense == 'le' else Domain.equalsTo(0.0)))
    return constraints

def AddHeadLossConstraints(model):
    for pipe_family in model._pipe_families.values():
        for t in model._set_of_time_periods:
            model.constraint(model._flow[model._link_index[pipe_family.label],t] == sum(pipe.flow_lower_bound[t] if pipe.fixed_flow[t] else (model._flow_pos[model._flow_pos_indices[pipe.label,t]] if (pipe.label,t) in model._flow_pos_indices else pipe.positive_flow_lower_bound[t] * model._flow_status_pos[model._pipe_index[pipe.label],t]) + (model._flow_neg[model._flow_neg_indices[pipe.label,t]] if (pipe.label,t) in model._flow_neg_indices else pipe.negative_flow_upper_bound[t] * model._flow_status_neg[model._pipe_index[pipe.label],t]) for pipe in pipe_family.options if pipe.selection_decision != 0))
            model.constraint(model._head[model._node_index[pipe_family.start_node.label],t] - sum((model._head_start_pos[model._positive_indices[pipe.label,t]] if (pipe.label,t) in model._positive_indices else 0.0) + (model._head_start_neg[model._negative_indices[pipe.label,t]] if (pipe.label,t) in model._negative_indices else 0.0) for pipe in pipe_family.options) >= model._head_lb.pick([[model._node_index[pipe_family.start_node.label],t]]) * (1 - sum(model._pipe_status[pipe.label] if model._problem == 'operational' else model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options)))
            model.constraint(model._head[model._node_index[pipe_family.start_node.label],t] - sum((model._head_start_pos[model._positive_indices[pipe.label,t]] if (pipe.label,t) in model._positive_indices else 0.0) + (model._head_start_neg[model._negative_indices[pipe.label,t]] if (pipe.label,t) in model._negative_indices else 0.0) for pipe in pipe_family.options) <= model._head_ub.pick([[model._node_index[pipe_family.start_node.label],t]]) * (1 - sum(model._pipe_status[pipe.label] if model._problem == 'operational' else model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options)))
            model.constraint(model._head[model._node_index[pipe_family.end_node.label],t] - sum((model._head_end_pos[model._positive_indices[pipe.label,t]] if (pipe.label,t) in model._positive_indices else 0.0) + (model._head_end_neg[model._negative_indices[pipe.label,t]] if (pipe.label,t) in model._negative_indices else 0.0) for pipe in pipe_family.options) >= model._head_lb.pick([[model._node_index[pipe_family.end_node.label],t]]) * (1 - sum(model._pipe_status[pipe.label] if model._problem == 'operational' else model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options)))
            model.constraint(model._head[model._node_index[pipe_family.end_node.label],t] - sum((model._head_end_pos[model._positive_indices[pipe.label,t]] if (pipe.label,t) in model._positive_indices else 0.0) + (model._head_end_neg[model._negative_indices[pipe.label,t]] if (pipe.label,t) in model._negative_indices else 0.0) for pipe in pipe_family.options) <= model._head_ub.pick([[model._node_index[pipe_family.end_node.label],t]]) * (1 - sum(model._pipe_status[pipe.label] if model._problem == 'operational' else model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options)))

    for i, pipe in enumerate(model._pipes.values()):
        if pipe.selection_decision == 0: continue
        for t in model._set_of_time_periods:
            model.constraint(model._flow_status_pos[i,t] + model._flow_status_neg[i,t] == (model._pipe_status[pipe.label] if model._problem == 'operational' else model._pipe_status[i]))
            if (pipe.label,t) in model._positive_indices:
                index = model._positive_indices[pipe.label,t]
                model.constraint(model._head_start_pos[index] >= model._head_lb.pick([[model._node_index[pipe.start_node.label],t]]) * model._flow_status_pos[i,t])
                model.constraint(model._head_start_pos[index] <= model._head_ub.pick([[model._node_index[pipe.start_node.label],t]]) * model._flow_status_pos[i,t])
                model.constraint(model._head_end_pos[index] >= model._head_lb.pick([[model._node_index[pipe.end_node.label],t]]) * model._flow_status_pos[i,t])
                model.constraint(model._head_end_pos[index] <= model._head_ub.pick([[model._node_index[pipe.end_node.label],t]]) * model._flow_status_pos[i,t])

                if pipe.positive_flow_lower_bound[t] == pipe.flow_upper_bound[t]:
                    model.constraint(model._head_start_pos[index] - model._head_end_pos[index] == pipe.resistance_coefficient * (pipe.positive_flow_lower_bound[t] ** pipe.exponent) * model._flow_status_pos[i,t])
                else:
                    model.constraint(model._flow_pos[model._flow_pos_indices[pipe.label,t]] >= model._arc_flow_lb_pos.pick([[model._arc_index[pipe.label],t]]) * model._flow_status_pos[i,t])
                    model.constraint(model._flow_pos[model._flow_pos_indices[pipe.label,t]] <= model._arc_flow_ub.pick([[model._arc_index[pipe.label],t]]) * model._flow_status_pos[i,t])
                    model.constraint(f'pipe_power_cone({pipe.label},{t},pos)', Expr.vstack(model._head_start_pos[index] - model._head_end_pos[index], model._flow_status_pos[i,t], (pipe.resistance_coefficient ** (1 / pipe.exponent)) * model._flow_pos[model._flow_pos_indices[pipe.label,t]]), Domain.inPPowerCone(1 / pipe.exponent))

            if (pipe.label,t) in model._negative_indices:
                index = model._negative_indices[pipe.label,t]
                model.constraint(model._head_start_neg[index] >= model._head_lb.pick([[model._node_index[pipe.start_node.label],t]]) * model._flow_status_neg[i,t])
                model.constraint(model._head_start_neg[index] <= model._head_ub.pick([[model._node_index[pipe.start_node.label],t]]) * model._flow_status_neg[i,t])
                model.constraint(model._head_end_neg[index] >= model._head_lb.pick([[model._node_index[pipe.end_node.label],t]]) * model._flow_status_neg[i,t])
                model.constraint(model._head_end_neg[index] <= model._head_ub.pick([[model._node_index[pipe.end_node.label],t]]) * model._flow_status_neg[i,t])
                if pipe.flow_lower_bound[t] == pipe.negative_flow_upper_bound[t]:
                    model.constraint(model._head_start_neg[index] - model._head_end_neg[index] == pipe.resistance_coefficient * np.sign(pipe.flow_lower_bound[t]) * abs(pipe.flow_lower_bound[t]) ** pipe.exponent * model._flow_status_neg[i,t])
                else:
                    model.constraint(model._flow_neg[model._flow_neg_indices[pipe.label,t]] >= model._arc_flow_lb.pick([[model._arc_index[pipe.label],t]]) * model._flow_status_neg[i,t])
                    model.constraint(model._flow_neg[model._flow_neg_indices[pipe.label,t]] <= model._arc_flow_ub_neg.pick([[model._arc_index[pipe.label],t]]) * model._flow_status_neg[i,t])
                    model.constraint(f'pipe_power_cone({pipe.label},{t},neg)', Expr.vstack(model._head_end_neg[index] - model._head_start_neg[index], model._flow_status_neg[i,t], (pipe.resistance_coefficient ** (1 / pipe.exponent)) * model._flow_neg[model._flow_neg_indices[pipe.label,t]]), Domain.inPPowerCone(1 / pipe.exponent))

def AddPipeFlowConstraints(model):
    if model._problem == 'operational': return
    for pipe_family in model._pipe_families.values():
        if pipe_family.is_optional:
            model.constraint(sum(model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options) <= 1)
        else:
            model.constraint(sum(model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options) == 1)
        for t in model._set_of_time_periods:
            model.constraint(model._flow[model._link_index[pipe_family.label],t] >= sum(model._arc_flow_lb.pick([[model._arc_index[pipe.label],t]]) * model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options))
            model.constraint(model._flow[model._link_index[pipe_family.label],t] <= sum(model._arc_flow_ub.pick([[model._arc_index[pipe.label],t]]) * model._pipe_status[model._pipe_index[pipe.label]] for pipe in pipe_family.options))

def AddValveFlowConstraints(model):
    for i, valve in enumerate(model._valves.values()):
        for t in model._set_of_time_periods:
            if valve.status_decisions[t] == -1:
                model.constraint(f'valve_flow_l_{valve.label}_{t}', model._flow[model._link_index[valve.label],t] >= model._arc_flow_lb_pos.pick([[model._arc_index[valve.label],t]]) * model._valve_status[i,t])
                model.constraint(f'valve_flow_u_{valve.label}_{t}', model._flow[model._link_index[valve.label],t] <= model._arc_flow_ub.pick([[model._arc_index[valve.label],t]]) * model._valve_status[i,t])

def AddPumpFlowConstraints(model):
    for i, pump in enumerate(model._pumps.values()):
        for t in model._set_of_time_periods:
            if pump.status_decisions[t] == -1:
                model.constraint(f'pump_flow_l_{pump.label}_{t}', model._flow[model._link_index[pump.label],t] >= model._arc_flow_lb_pos.pick([[model._arc_index[pump.label],t]]) * model._pump_status[i,t])
                model.constraint(f'pump_flow_u_{pump.label}_{t}', model._flow[model._link_index[pump.label],t] <= model._arc_flow_ub.pick([[model._arc_index[pump.label],t]]) * model._pump_status[i,t])

def AddPumpHeadConstraints(model):
    for i, pump in enumerate(model._pumps.values()):
        for t in model._set_of_time_periods:
            if pump.status_decisions[t] == 0: continue
            if pump.fixed_flow[t]:
                head_gain_flow = pump.curve_b * (pump.flow_lower_bound[t] ** pump.curve_c)
            elif pump.curve_c == 1.0:
                head_gain_flow = pump.curve_b * model._flow[model._link_index[pump.label],t]
            else:
                index = model._positive_indices[pump.label,t]
                model.constraint(model._head_start_pos[index] >= model._head_lb.pick([[model._node_index[pump.start_node.label],t]]) * model._pump_status[i,t])
                model.constraint(model._head_start_pos[index] <= model._head_ub.pick([[model._node_index[pump.start_node.label],t]]) * model._pump_status[i,t])
                model.constraint(model._head[model._node_index[pump.start_node.label],t] - model._head_start_pos[index] >= model._head_lb.pick([[model._node_index[pump.start_node.label],t]]) * (1 - model._pump_status[i,t]))
                model.constraint(model._head[model._node_index[pump.start_node.label],t] - model._head_start_pos[index] <= model._head_ub.pick([[model._node_index[pump.start_node.label],t]]) * (1 - model._pump_status[i,t]))
                model.constraint(model._head_end_pos[index] >= model._head_lb.pick([[model._node_index[pump.end_node.label],t]]) * model._pump_status[i,t])
                model.constraint(model._head_end_pos[index] <= model._head_ub.pick([[model._node_index[pump.end_node.label],t]]) * model._pump_status[i,t])
                model.constraint(model._head[model._node_index[pump.end_node.label],t] - model._head_end_pos[index] >= model._head_lb.pick([[model._node_index[pump.end_node.label],t]]) * (1 - model._pump_status[i,t]))
                model.constraint(model._head[model._node_index[pump.end_node.label],t] - model._head_end_pos[index] <= model._head_ub.pick([[model._node_index[pump.end_node.label],t]]) * (1 - model._pump_status[i,t]))
                if pump.positive_flow_lower_bound[t] == pump.flow_upper_bound[t]:
                    model.constraint(model._head_start_pos[index] - model._head_end_pos[index] + (pump.curve_a + pump.curve_b * (pump.positive_flow_lower_bound[t] ** pump.curve_c)) * model._pump_status[i,t] == 0)
                else:
                    model.constraint(f'pump_power_cone({pump.label},{t})', Expr.vstack(model._head_start_pos[index] - model._head_end_pos[index] + pump.curve_a * model._pump_status[i,t], model._pump_status[i,t], (-pump.curve_b) ** (1 / pump.curve_c) * model._flow[model._link_index[pump.label],t]), Domain.inPPowerCone(1 / pump.curve_c))
                continue
            if pump.status_decisions[t] == 1:
                model.constraint(model._head[model._node_index[pump.start_node.label],t] - model._head[model._node_index[pump.end_node.label],t] + pump.curve_a + head_gain_flow == 0)
            else:
                model.constraint(model._head[model._node_index[pump.start_node.label],t] - model._head[model._node_index[pump.end_node.label],t] + pump.curve_a * model._pump_status[i,t] + head_gain_flow >= model._head_lb.pick([[model._node_index[pump.start_node.label],t]]) * (1 - model._pump_status[i,t]) - model._head_ub.pick([[model._node_index[pump.end_node.label],t]]) * (1 - model._pump_status[i,t]))
                model.constraint(model._head[model._node_index[pump.start_node.label],t] - model._head[model._node_index[pump.end_node.label],t] + pump.curve_a * model._pump_status[i,t] + head_gain_flow <= model._head_ub.pick([[model._node_index[pump.start_node.label],t]]) * (1 - model._pump_status[i,t]) - model._head_lb.pick([[model._node_index[pump.end_node.label],t]]) * (1 - model._pump_status[i,t]))

def AddValveHeadConstraints(model):
    for i, valve in enumerate(model._valves.values()):
        for t in model._set_of_time_periods:
            if valve.status_decisions[t] == 0: continue
            if valve.fixed_flow[t]:
                if valve.status_decisions[t] == 1:
                    model.constraint(model._head[model._node_index[valve.start_node.label],t] - model._head[model._node_index[valve.end_node.label],t] == valve.resistance_coefficient * (valve.flow_lower_bound[t] ** valve.exponent))
                else:
                    model.constraint(model._head[model._node_index[valve.start_node.label],t] - model._head[model._node_index[valve.end_node.label],t] - valve.resistance_coefficient * (valve.flow_lower_bound[t] ** valve.exponent) >= model._head_lb.pick([[model._node_index[valve.start_node.label],t]]) * (1 - model._valve_status[i,t]) - model._head_ub.pick([[model._node_index[valve.end_node.label],t]]) * (1 - model._valve_status[i,t]))
                    model.constraint(model._head[model._node_index[valve.start_node.label],t] - model._head[model._node_index[valve.end_node.label],t] - valve.resistance_coefficient * (valve.flow_lower_bound[t] ** valve.exponent) <= model._head_ub.pick([[model._node_index[valve.start_node.label],t]]) * (1 - model._valve_status[i,t]) - model._head_lb.pick([[model._node_index[valve.end_node.label],t]]) * (1 - model._valve_status[i,t]))
            else:
                index = model._positive_indices[valve.label,t]
                model.constraint(model._head_start_pos[index] >= model._head_lb.pick([[model._node_index[valve.start_node.label],t]]) * model._valve_status[i,t])
                model.constraint(model._head_start_pos[index] <= model._head_ub.pick([[model._node_index[valve.start_node.label],t]]) * model._valve_status[i,t])
                model.constraint(model._head[model._node_index[valve.start_node.label],t] - model._head_start_pos[index] >= model._head_lb.pick([[model._node_index[valve.start_node.label],t]]) * (1 - model._valve_status[i,t]))
                model.constraint(model._head[model._node_index[valve.start_node.label],t] - model._head_start_pos[index] <= model._head_ub.pick([[model._node_index[valve.start_node.label],t]]) * (1 - model._valve_status[i,t]))
                model.constraint(model._head_end_pos[index] >= model._head_lb.pick([[model._node_index[valve.end_node.label],t]]) * model._valve_status[i,t])
                model.constraint(model._head_end_pos[index] <= model._head_ub.pick([[model._node_index[valve.end_node.label],t]]) * model._valve_status[i,t])
                model.constraint(model._head[model._node_index[valve.end_node.label],t] - model._head_end_pos[index] >= model._head_lb.pick([[model._node_index[valve.end_node.label],t]]) * (1 - model._valve_status[i,t]))
                model.constraint(model._head[model._node_index[valve.end_node.label],t] - model._head_end_pos[index] <= model._head_ub.pick([[model._node_index[valve.end_node.label],t]]) * (1 - model._valve_status[i,t]))
                if valve.positive_flow_lower_bound[t] == valve.flow_upper_bound[t]:
                    model.constraint(model._head_start_pos[index] - model._head_end_pos[index] == valve.resistance_coefficient * (valve.positive_flow_lower_bound[t] ** valve.exponent) * model._valve_status[i,t])
                else:
                    model.constraint(f'valve_power_cone({valve.label},{t})', Expr.vstack(model._head_start_pos[index] - model._head_end_pos[index], model._valve_status[i,t], (valve.resistance_coefficient ** (1 / valve.exponent)) * model._flow[model._link_index[valve.label],t]), Domain.inPPowerCone(1 / valve.exponent))

def AddTankBalanceConstraints(model):
    for i, tank in enumerate(model._tanks.values()):
        incidence = Matrix.sparse(1, len(model._set_of_links), [0] * (len(model._incoming_edges[tank.label]) + len(model._outgoing_edges[tank.label])), [model._link_index[link] for link in model._incoming_edges[tank.label]] + [model._link_index[link] for link in model._outgoing_edges[tank.label]], [1.0] * len(model._incoming_edges[tank.label]) + [-1.0] * len(model._outgoing_edges[tank.label]))
        model.constraint(f'tank_balance({tank.label})', tank.tank_coef * Expr.mul(incidence, model._flow) == (Expr.hstack(model._head[model._node_index[tank.label],1:], model._tank_terminal_head[i]) if model._time_steps > 1 else model._tank_terminal_head[i]) - model._head[model._node_index[tank.label],:])

def AddJunctionBalanceConstraints(model):
    for junction in model._junctions.values():
        if junction.segment == 'reservoir': continue
        incidence = Matrix.sparse(1, len(model._set_of_links), [0] * (len(model._incoming_edges[junction.label]) + len(model._outgoing_edges[junction.label])), [model._link_index[link] for link in model._incoming_edges[junction.label]] + [model._link_index[link] for link in model._outgoing_edges[junction.label]], [1.0] * len(model._incoming_edges[junction.label]) + [-1.0] * len(model._outgoing_edges[junction.label]))
        model.constraint(f'junction_balance({junction.label})', Expr.mul(incidence, model._flow) == np.array([junction.net_demand[:model._time_steps]]))

def AddValidInequalities(model, valid_inequalities):
    model._valid_cons = []
    indices = {'pump_status': model._pump_index, 'valve_status': model._valve_index, 'flow_status_pos': model._pipe_index, 'flow_status_neg': model._pipe_index}
    for sense, terms, rhs in sorted(valid_inequalities):
        expression = [Expr.constTerm(0.0)]
        for (variable, label, period), coefficient in terms:
            if variable == 'pipe_status':
                value = model._pipe_status[label] if model._problem == 'operational' else model._pipe_status[model._pipe_index[label]]
            else:
                value = getattr(model, '_' + variable)[indices[variable][label],period]
            expression.append(float(coefficient) * (value if isinstance(value, Expression) else Expr.constTerm(float(value))))
        model._valid_cons.append(model.constraint(Expr.add(expression), Domain.greaterThan(float(rhs)) if sense == 'ge' else Domain.lessThan(float(rhs)) if sense == 'le' else Domain.equalsTo(float(rhs))))

def AddNoGoodCutMosek(model, previous_solution, up_to=None):
    up_to = model._time_steps if up_to is None else up_to
    expression = []
    pump_incumbent = np.array([previous_solution['pump_status'][pump][:up_to] for pump in model._set_of_pumps], dtype=int).reshape(-1)
    valve_incumbent = np.array([previous_solution['valve_status'][valve][:up_to] for valve in model._set_of_valves], dtype=int).reshape(-1)
    pipe_incumbent = np.array([previous_solution['pipe_status'][pipe] for pipe in model._set_of_pipes], dtype=int) if model._problem != 'operational' else np.array([], dtype=int)

    if len(pump_incumbent) > 0:
        expression.append(Expr.dot((2 * pump_incumbent - 1).astype(float), Var.flatten(model._pump_status[:, :up_to])))
    if len(valve_incumbent) > 0:
        expression.append(Expr.dot((2 * valve_incumbent - 1).astype(float), Var.flatten(model._valve_status[:, :up_to])))
    if len(pipe_incumbent) > 0:
        expression.append(Expr.dot((2 * pipe_incumbent - 1).astype(float), model._pipe_status))
    if expression:
        model.constraint(Expr.add(expression) <= float(np.sum(pump_incumbent) + np.sum(valve_incumbent) + np.sum(pipe_incumbent) - 1))

def MosekModel(model_name, network, binary_flow_direction=True, binary_pump_valve=True, binary_pipe_selection=False):
    model = Model(model_name)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps, model._problem = network.junctions, network.tanks, network.pipes, network.pumps, network.valves, network.time_steps, network.problem
    model._nodes, model._arcs, model._links, model._pipe_families = network.junctions | network.tanks, network.pipes | network.pumps | network.valves, network.pipe_families | network.pumps | network.valves, network.pipe_families
    model._binary_flow_direction, model._binary_pump_valve, model._binary_pipe_selection = binary_flow_direction, binary_pump_valve, binary_pipe_selection

    DefineSets(model)
    DefineBounds(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model)
    AddPipeFlowConstraints(model)
    AddPipeFacets(model)

    if model._set_of_pumps:
        AddPumpFlowConstraints(model)
        AddPumpHeadConstraints(model)
        AddPumpFacets(model)

    if model._set_of_valves:
        AddValveFlowConstraints(model)
        AddValveHeadConstraints(model)
        AddValveFacets(model)

    if model._set_of_tanks:
        AddTankBalanceConstraints(model)

    if model._set_of_junctions:
        AddJunctionBalanceConstraints(model)

    if getattr(network, 'valid_inequalities', None):
        AddValidInequalities(model, network.valid_inequalities)

    #model.setSolverParam('ptfWriteSolutions', 'on')
    model.setSolverParam('numThreads', 12)

    model._is_mip = bool((model._pipes and (binary_flow_direction or (model._problem != 'operational' and binary_pipe_selection))) or (binary_pump_valve and (model._pumps or model._valves)))
    if model._is_mip:
        model.setSolverParam('mioMaxTime', network.time_limit)
    else:
        model.setSolverParam('optimizerMaxTime', network.time_limit)

    model.acceptedSolutionStatus(AccSolutionStatus.Feasible)

    return model

def SolveMosekModel(model, model_name, results_directory):
    with open(os.path.join(results_directory, f'{model_name}_log.log'), 'w') as f:
        model.setLogHandler(f)
        model.solve()
    model.setLogHandler(None)
    #model.writeTask(os.path.join(results_directory, f'{model_name}_solution.ptf'))

    problem_status = 'feasible' if model.getPrimalSolutionStatus() in (SolutionStatus.Optimal, SolutionStatus.Feasible) else 'infeasible' if model.getProblemStatus() == ProblemStatus.PrimalInfeasible else 'nosolution'

    lower_bound = None
    if problem_status != 'infeasible':
        if model._is_mip:
            lower_bound = model.getSolverDoubleInfo('mioObjBound') if model.getSolverIntInfo('mioObjBoundDefined') else None
        elif model.getDualSolutionStatus() in (SolutionStatus.Optimal, SolutionStatus.Feasible):
            lower_bound = model.dualObjValue()

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if model._set_of_nodes:
            head_levels = np.array(model._head.level()).reshape(len(model._set_of_nodes), len(model._set_of_time_periods))
            solution['head'] = {node: head_levels[i] for i, node in enumerate(model._set_of_nodes)}

        if model._set_of_tanks:
            terminal_head_levels = np.array(model._tank_terminal_head.level()).reshape(len(model._set_of_tanks))
            for i, tank in enumerate(model._set_of_tanks):
                solution['head'][tank] = np.append(solution['head'][tank], terminal_head_levels[i])

        if model._set_of_links:
            flow_solution_levels = np.array(model._flow.level()).reshape(len(model._set_of_links), len(model._set_of_time_periods))
            solution['flow'] = {link: flow_solution_levels[i] for i, link in enumerate(model._set_of_links)}

        if model._set_of_pumps:
            pump_status_solution_levels = (np.rint(model._pump_status.level()) if model._binary_pump_valve else np.array(model._pump_status.level())).reshape(len(model._set_of_pumps), len(model._set_of_time_periods))
            solution['pump_status'] = {pump: pump_status_solution_levels[i] for i, pump in enumerate(model._set_of_pumps)}

        if model._set_of_valves:
            valve_status_solution_levels = (np.rint(model._valve_status.level()) if model._binary_pump_valve else np.array(model._valve_status.level())).reshape(len(model._set_of_valves), len(model._set_of_time_periods))
            solution['valve_status'] = {valve: valve_status_solution_levels[i] for i, valve in enumerate(model._set_of_valves)}

        if model._set_of_pipes:
            pipe_status_solution_levels = (np.rint(model._pipe_status.level()) if model._binary_pipe_selection or model._binary_flow_direction else np.array(model._pipe_status.level())) if model._problem != 'operational' else np.ones(len(model._set_of_pipes))
            solution['pipe_status'] = {pipe: int(pipe_status_solution_levels[i]) if model._problem == 'operational' or model._binary_pipe_selection or model._binary_flow_direction else float(pipe_status_solution_levels[i]) for i, pipe in enumerate(model._set_of_pipes)}

    return problem_status, solution, lower_bound
