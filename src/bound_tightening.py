import os
import time
import numpy as np
import concurrent.futures
from mosek.fusion import *
import mosek.fusion.pythonic
from .helpers import round_bound, write_results
from .read_epanet import set_timesteps
from .hull_facets import find_hydraulic_loss_facets, find_head_gain_facets
from .models.mosek import MosekModel

def get_mosek_bound(model, is_mip):
    try:
        if model.getProblemStatus() == ProblemStatus.PrimalInfeasible: return 'infeasible'
        if is_mip:
            if not model.getSolverIntInfo('mioObjBoundDefined'): return 'unknown'
            bound = model.getSolverDoubleInfo('mioObjBound')
        else:
            if model.getDualSolutionStatus() not in (SolutionStatus.Optimal, SolutionStatus.Feasible): return 'unknown'
            bound = model.dualObjValue()
        return bound if np.isfinite(bound) else 'unknown'
    except (OptimizeError, SolutionError):
        return 'unknown'

def solve_mosek_problem(model, senses, expressions, fixings=None, batch_workers=1):
    if not expressions: return []
    clones = []
    try:
        for i, (sense, expression) in enumerate(zip(senses, expressions)):
            cloned_model = model.clone()
            clones.append(cloned_model)
            cloned_model.objective('objective', sense, expression)
            if fixings is not None:
                cloned_model.constraint(fixings[i][0], Domain.equalsTo(fixings[i][1]))
            cloned_model.setSolverParam('numThreads', 1)
            cloned_model.setSolverParam('intpntSolveForm', 'primal')
        try:
            statuses = Model.solveBatch(False, -1.0, batch_workers, clones)
        except (OptimizeError, SolutionError):
            statuses = None
        return [get_mosek_bound(clone, model._is_mip) if statuses is None or statuses[i] == SolverStatus.OK else 'unknown' for i, clone in enumerate(clones)]
    finally:
        for clone in clones: clone.dispose()

def solve_variable_bounds(model, expressions, bounds, epsilon, batch_workers):
    for sense in (ObjectiveSense.Minimize, ObjectiveSense.Maximize):
        values = solve_mosek_problem(model, [sense] * len(bounds), expressions, batch_workers=batch_workers)
        for i, value in enumerate(values):
            if value in ('unknown', 'infeasible'): continue
            if sense == ObjectiveSense.Minimize:
                candidate = round_bound(value - epsilon * max(abs(value), 1.0), 'floor')
                bounds[i][0] = max(bounds[i][0], candidate)
            else:
                candidate = round_bound(value + epsilon * max(abs(value), 1.0), 'ceiling')
                bounds[i][1] = min(bounds[i][1], candidate)
    return bounds

def solve_conditional_variable_bounds(model, expressions, bounds, statuses, senses, epsilon, batch_workers):
    values = solve_mosek_problem(model, senses, expressions, [(status, 1.0) for status in statuses], batch_workers=batch_workers)
    for i, value in enumerate(values):
        if value == 'infeasible':
            bounds[i] = 'infeasible'
            continue
        if value == 'unknown': continue
        if senses[i] == ObjectiveSense.Minimize:
            candidate = round_bound(value - epsilon * max(abs(value), 1.0), 'floor')
            bounds[i] = max(bounds[i], candidate)
        else:
            candidate = round_bound(value + epsilon * max(abs(value), 1.0), 'ceiling')
            bounds[i] = min(bounds[i], candidate)
    return bounds

def solve_variable_binary(model, expressions, state, batch_workers=1):
    values = solve_mosek_problem(model, [ObjectiveSense.Minimize] * len(expressions), [Expr.constTerm(0.0) for _ in expressions], [(expression, state) for expression in expressions], batch_workers=batch_workers)
    return [int(1 - state) if value == 'infeasible' else -1 for value in values]

def get_bounds(network, horizon=None):
    start, end = (0, network.time_steps) if horizon is None else horizon
    nodes, arcs = network.junctions | network.tanks, network.pipes | network.pumps | network.valves
    return {'flow': {label: np.array([arc.flow_lower_bound[start:end], arc.flow_upper_bound[start:end]]) for label, arc in arcs.items()},
            'head': {label: np.array([node.head_lower_bound[start:end + (label in network.tanks)], node.head_upper_bound[start:end + (label in network.tanks)]]) for label, node in nodes.items()},
            'positive_flow_lb': {label: arc.positive_flow_lower_bound[start:end].copy() for label, arc in arcs.items()},
            'negative_flow_ub': {label: pipe.negative_flow_upper_bound[start:end].copy() for label, pipe in network.pipes.items()},
            'pipe_status': {label: pipe.selection_decision for label, pipe in network.pipes.items()},
            'flow_direction': {label: pipe.direction_decision[start:end].copy() for label, pipe in network.pipes.items()},
            'pump_status': {label: pump.status_decisions[start:end].copy() for label, pump in network.pumps.items()},
            'valve_status': {label: valve.status_decisions[start:end].copy() for label, valve in network.valves.items()}}

def update_bounds(network, results):
    nodes, arcs = network.junctions | network.tanks, network.pipes | network.pumps | network.valves
    for start, bounds in results.items():
        for label, status in bounds['pipe_status'].items():
            if status == -1: continue
            network.pipes[label].selection_decision = status
        for variable, components, attribute in [('flow_direction', network.pipes, 'direction_decision'), ('pump_status', network.pumps, 'status_decisions'), ('valve_status', network.valves, 'status_decisions')]:
            for label, values in bounds[variable].items():
                decisions = getattr(components[label], attribute)[start:start + len(values)]
                known = values != -1
                if variable == 'flow_direction' and network.pipes[label].selection_decision == 0: continue
                decisions[known] = values[known]
        for variable, components in [('head', nodes), ('flow', arcs)]:
            for label, values in bounds[variable].items():
                lower = getattr(components[label], variable + '_lower_bound')[start:start + values.shape[1]]
                upper = getattr(components[label], variable + '_upper_bound')[start:start + values.shape[1]]
                np.maximum(lower, values[0], out=lower)
                np.minimum(upper, values[1], out=upper)
        for label, values in bounds['positive_flow_lb'].items():
            lower = arcs[label].positive_flow_lower_bound[start:start + len(values)]
            np.maximum(lower, values, out=lower)
        for label, values in bounds['negative_flow_ub'].items():
            upper = network.pipes[label].negative_flow_upper_bound[start:start + len(values)]
            np.minimum(upper, values, out=upper)

    for arc in arcs.values():
        arc.add_flow_bounds()

    return get_bounds(network)

def perform_obbt(network, period, horizon, binary_flow_direction=False, binary_pump_valve=False, binary_pipe_selection=False, batch_workers=1, epsilon=1e-4):
    start, end = horizon
    set_timesteps(network.junctions, network.tanks, network.pipes, network.pumps, network.valves, list(range(start, end)))
    network.time_steps, t = end - start, period - start
    find_hydraulic_loss_facets(network.pipes | network.valves, network.time_steps)
    find_head_gain_facets(network.pumps, network.time_steps)

    model = MosekModel(model_name=f'obbt_{period}', network=network, binary_flow_direction=binary_flow_direction, binary_pump_valve=binary_pump_valve, binary_pipe_selection=binary_pipe_selection)
    bounds = get_bounds(network, (t, t+1))

    try:
        pipe_flows = {}
        for label, pipe in model._pipes.items():
            if pipe.selection_decision == 0: continue
            positive_flow = model._flow_pos[model._flow_pos_indices[label,t]] if (label,t) in model._flow_pos_indices else pipe.positive_flow_lower_bound[t] * model._flow_status_pos[model._pipe_index[label],t]
            negative_flow = model._flow_neg[model._flow_neg_indices[label,t]] if (label,t) in model._flow_neg_indices else pipe.negative_flow_upper_bound[t] * model._flow_status_neg[model._pipe_index[label],t]
            pipe_flows[label] = positive_flow, negative_flow

        labels = [label for label in pipe_flows if bounds['flow'][label][1,0] - bounds['flow'][label][0,0] > epsilon]
        for batch_start in range(0, len(labels), batch_workers):
            batch = labels[batch_start:batch_start + batch_workers]
            expressions = [sum(pipe_flows[label]) for label in batch]
            new_bounds = solve_variable_bounds(model, expressions, [bounds['flow'][label][:,0].tolist() for label in batch], epsilon, batch_workers)
            for label, expression, (lower, upper) in zip(batch, expressions, new_bounds):
                prev_lb, prev_ub = bounds['flow'][label][:,0]
                if lower > prev_lb: model.constraint(expression >= lower)
                if upper < prev_ub: model.constraint(expression <= upper)
                bounds['flow'][label][:,0] = lower, upper
                bounds['positive_flow_lb'][label][0] = max(bounds['positive_flow_lb'][label][0], lower)
                bounds['negative_flow_ub'][label][0] = min(bounds['negative_flow_ub'][label][0], upper)
                if lower > 0.0 or upper < 0.0:
                    if bounds['pipe_status'][label] != 1 or bounds['flow_direction'][label][0] == -1:
                        flow_status = model._flow_status_pos if lower > 0.0 else model._flow_status_neg
                        model.constraint(flow_status[model._pipe_index[label],t], Domain.equalsTo(1.0))
                    bounds['pipe_status'][label] = 1
                    bounds['flow_direction'][label][0] = int(lower > 0.0)

        for variable, arcs, indices in [('pump_status', model._pumps, model._pump_index), ('valve_status', model._valves, model._valve_index)]:
            labels = [label for label in arcs if bounds[variable][label][0] != 0 and bounds['flow'][label][1,0] - bounds['flow'][label][0,0] > epsilon]
            for batch_start in range(0, len(labels), batch_workers):
                batch = labels[batch_start:batch_start + batch_workers]
                expressions = [model._flow[model._link_index[label],t] for label in batch]
                new_bounds = solve_variable_bounds(model, expressions, [bounds['flow'][label][:,0].tolist() for label in batch], epsilon, batch_workers)
                for label, expression, (lower, upper) in zip(batch, expressions, new_bounds):
                    prev_lb, prev_ub = bounds['flow'][label][:,0]
                    if lower > prev_lb: model.constraint(expression >= lower)
                    if upper < prev_ub: model.constraint(expression <= upper)
                    bounds['flow'][label][:,0] = lower, upper
                    bounds['positive_flow_lb'][label][0] = max(bounds['positive_flow_lb'][label][0], lower)
                    if lower > 0.0 and bounds[variable][label][0] == -1:
                        model.constraint(getattr(model, '_' + variable)[indices[label],t], Domain.equalsTo(1.0))
                        bounds[variable][label][0] = 1

        head_variables = [(label, 0, model._head[model._node_index[label],t]) for label in model._nodes if bounds['head'][label][1,0] - bounds['head'][label][0,0] > epsilon]
        head_variables.extend((label, 1, model._head[model._node_index[label],t+1] if t+1 < model._time_steps else model._tank_terminal_head[model._tank_index[label]]) for label in model._tanks if bounds['head'][label][1,1] - bounds['head'][label][0,1] > epsilon)
        for batch_start in range(0, len(head_variables), batch_workers):
            batch = head_variables[batch_start:batch_start + batch_workers]
            new_bounds = solve_variable_bounds(model, [expression for _, _, expression in batch], [bounds['head'][label][:,offset].tolist() for label, offset, _ in batch], epsilon, batch_workers)
            for (label, offset, expression), (lower, upper) in zip(batch, new_bounds):
                prev_lb, prev_ub = bounds['head'][label][:,offset]
                if lower > prev_lb: model.constraint(expression >= lower)
                if upper < prev_ub: model.constraint(expression <= upper)
                bounds['head'][label][:,offset] = lower, upper

        for positive, sense, bound_variable, flow_status in [(True, ObjectiveSense.Minimize, 'positive_flow_lb', model._flow_status_pos), (False, ObjectiveSense.Maximize, 'negative_flow_ub', model._flow_status_neg)]:
            labels = [label for label in pipe_flows if bounds['pipe_status'][label] != 0 and (bounds['flow_direction'][label][0] == -1 or (bounds['pipe_status'][label] == -1 and bounds['flow_direction'][label][0] == int(positive)))]
            for batch_start in range(0, len(labels), batch_workers):
                batch = labels[batch_start:batch_start + batch_workers]
                expressions = [pipe_flows[label][0 if positive else 1] for label in batch]
                statuses = [flow_status[model._pipe_index[label],t] for label in batch]
                values = solve_conditional_variable_bounds(model, expressions, [bounds[bound_variable][label][0] for label in batch], statuses, [sense] * len(batch), epsilon, batch_workers)
                for label, expression, status, value in zip(batch, expressions, statuses, values):
                    if value != 'infeasible':
                        if positive and value > bounds[bound_variable][label][0]: model.constraint(expression >= value * status)
                        if not positive and value < bounds[bound_variable][label][0]: model.constraint(expression <= value * status)
                        bounds[bound_variable][label][0] = value
                    else:
                        model.constraint(status, Domain.equalsTo(0.0))
                        if bounds['flow_direction'][label][0] == int(positive):
                            bounds['pipe_status'][label] = 0
                        else:
                            bounds['flow_direction'][label][0] = 1 - int(positive)

        for variable, arcs, indices in [('pump_status', model._pumps, model._pump_index), ('valve_status', model._valves, model._valve_index)]:
            arc_labels = [label for label in arcs if bounds[variable][label][0] == -1]
            for batch_start in range(0, len(arc_labels), batch_workers):
                batch = arc_labels[batch_start:batch_start + batch_workers]
                expressions = [model._flow[model._link_index[label],t] for label in batch]
                statuses = [getattr(model, '_' + variable)[indices[label],t] for label in batch]
                values = solve_conditional_variable_bounds(model, expressions, [bounds['positive_flow_lb'][label][0] for label in batch], statuses, [ObjectiveSense.Minimize] * len(batch), epsilon, batch_workers)
                for label, expression, status, value in zip(batch, expressions, statuses, values):
                    if value == 'infeasible':
                        model.constraint(status, Domain.equalsTo(0.0))
                        bounds[variable][label][0] = 0
                    else:
                        if value > bounds['positive_flow_lb'][label][0]: model.constraint(expression >= value * status)
                        bounds['positive_flow_lb'][label][0] = value

            arc_labels = [label for label in arcs if bounds[variable][label][0] == -1]
            for batch_start in range(0, len(arc_labels), batch_workers):
                batch = arc_labels[batch_start:batch_start + batch_workers]
                statuses = [getattr(model, '_' + variable)[indices[label],t] for label in batch]
                decisions = solve_variable_binary(model, statuses, state=0.0, batch_workers=batch_workers)
                for label, decision in zip(batch, decisions):
                    if decision == -1: continue
                    model.constraint(getattr(model, '_' + variable)[indices[label],t], Domain.equalsTo(float(decision)))
                    bounds[variable][label][0] = decision

        if model._problem != 'operational':
            for state in (1.0, 0.0):
                labels = [label for label in model._pipes if bounds['pipe_status'][label] == -1]
                for batch_start in range(0, len(labels), batch_workers):
                    batch = labels[batch_start:batch_start + batch_workers]
                    decisions = solve_variable_binary(model, [model._pipe_status[model._pipe_index[label]] for label in batch], state=state, batch_workers=batch_workers)
                    for label, decision in zip(batch, decisions):
                        if decision == -1: continue
                        model.constraint(model._pipe_status[model._pipe_index[label]], Domain.equalsTo(float(decision)))
                        bounds['pipe_status'][label] = decision
        return bounds
    finally:
        model.dispose()

def compare_bounds(earlier_bounds, current_bounds, epsilon=1e-4):
    shrinkage, statuses_fixed, unchanged = {'flow': [], 'head': []}, {}, True
    for variable, components in earlier_bounds.items():
        continuous = variable in ('flow', 'head', 'positive_flow_lb', 'negative_flow_ub')
        if not continuous: statuses_fixed[variable] = 0
        for label, earlier in components.items():
            current = current_bounds[variable][label]
            if variable in shrinkage:
                width = earlier[1] - earlier[0]
                unfixed = np.isfinite(width) & (width > 1e-9)
                current_width = current[1] - current[0]
                shrinkage[variable].extend(1 - current_width[unfixed] / width[unfixed])
            if continuous:
                unchanged &= np.allclose(earlier, current, atol=epsilon, rtol=epsilon)
            else:
                statuses_fixed[variable] += np.count_nonzero((earlier == -1) & (current != -1))
                unchanged &= np.array_equal(earlier, current)
    shrinkage = {variable: 100 * float(np.mean(values)) if values else 0.0 for variable, values in shrinkage.items()}
    return shrinkage, statuses_fixed, unchanged

def mosek_obbt(network, horizons, num_rounds=20, binary_flow_direction=False, binary_pump_valve=False, binary_pipe_selection=False, mode='w'):
    write_results(network.results_directory, 'obbt_results.txt', f'OBBT horizons: {horizons}', mode=mode)
    earlier_bounds = get_bounds(network)
    num_cpus = os.process_cpu_count() or 1
    num_workers = min(num_cpus, len(horizons))
    batch_workers = num_cpus // num_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for r in range(num_rounds):
            round_start = time.time()
            futures = {period: executor.submit(perform_obbt, network, period, horizon, binary_flow_direction, binary_pump_valve, binary_pipe_selection, batch_workers) for period, horizon in enumerate(horizons)}
            current_bounds = update_bounds(network, {period: future.result() for period, future in futures.items()})

            shrinkage, statuses_fixed, unchanged = compare_bounds(earlier_bounds, current_bounds)
            status_summary = ' | '.join(f'{variable} fixed : {count}' for variable, count in statuses_fixed.items())
            write_results(network.results_directory, 'obbt_results.txt', f'Round {r+1} time : {time.time() - round_start:.2f}')
            write_results(network.results_directory, 'obbt_results.txt', f"flow : {shrinkage['flow']:.2f}% | head : {shrinkage['head']:.2f}% | {status_summary}")
            write_results(network.results_directory, 'obbt_results.txt', '-------------------------------------------')

            if unchanged: break
            earlier_bounds = current_bounds
