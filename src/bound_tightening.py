import time
import numpy as np
import concurrent.futures
from mosek.fusion import *
import mosek.fusion.pythonic
from .read_epanet import set_timesteps
from .hull_facets import find_hydraulic_loss_facets, find_head_gain_facets
from .helpers import round_bound, write_bounds, write_results, valid_inequality_key
from .operational.mosek import MosekOperationalModel, AddPipeFacets, AddValveFacets, AddPumpFacets

_cached_worker_model = None

def init_obbt(model_name, results_directory, junctions, tanks, pipes, pumps, valves, horizon, hydraulic_timestep, time_limit, binary_pipe=False, binary_pump_valve=False):
    global _cached_worker_model

    start, end = horizon
    num_periods = end - start

    set_timesteps(junctions, tanks, pipes, pumps, valves, list(range(start, end)))
    find_hydraulic_loss_facets(pipes, num_periods)
    find_hydraulic_loss_facets(valves, num_periods)
    find_head_gain_facets(pumps, num_periods)

    _cached_worker_model = MosekOperationalModel(model_name, results_directory, junctions, tanks, pipes, pumps, valves, num_periods, hydraulic_timestep, time_limit, binary_pipe, binary_pump_valve, True)
    _cached_worker_model.setSolverParam("numThreads", 1)
    _cached_worker_model.setSolverParam("intpntSolveForm", "primal")
    _cached_worker_model._obbt_bound_cons = {}

def solve_mosek_problem(sense, expression, binary):
    _cached_worker_model.objective('objective', sense, expression)
    try:
        _cached_worker_model.solve()
    except (OptimizeError, SolutionError):
        return "unknown"

    if _cached_worker_model.getProblemStatus() == ProblemStatus.PrimalInfeasible: return "infeasible" #in (ProblemStatus.PrimalInfeasible, ProblemStatus.PrimalAndDualInfeasible)

    try:
        if binary:
            if not _cached_worker_model.getSolverIntInfo("mioObjBoundDefined"):
                return "unknown"
            bound = _cached_worker_model.getSolverDoubleInfo("mioObjBound")
        else:
            dual_status = _cached_worker_model.getDualSolutionStatus()
            if dual_status not in (SolutionStatus.Optimal, SolutionStatus.Feasible): return "unknown"
            _cached_worker_model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
            bound = _cached_worker_model.dualObjValue()

        return bound if np.isfinite(bound) else "unknown"

    except SolutionError:
        return "unknown"

def solve_variable_bounds(expression, lower, upper, binary, label, epsilon=1e-4):
    if upper - lower <= epsilon:
        return [lower, upper]

    value = solve_mosek_problem(ObjectiveSense.Minimize, expression, binary)
    if value not in ("unknown", "infeasible"):
        candidate = round_bound(value - epsilon * max(abs(value), 1), "floor")
        lower, upper = adjust_variable_bounds(expression, candidate, upper, lower, upper, label)

    value = solve_mosek_problem(ObjectiveSense.Maximize, expression, binary)
    if value not in ("unknown", "infeasible"):
        candidate = round_bound(value + epsilon * max(abs(value), 1), "ceiling")
        lower, upper = adjust_variable_bounds(expression, lower, candidate, lower, upper, label)

    return [lower, upper]

def solve_variable_binary(expression, binary):
    infeasible_states = []

    for state in (1, 0):
        cons = _cached_worker_model.constraint(expression, Domain.equalsTo(float(state)))
        try:
            value = solve_mosek_problem(ObjectiveSense.Minimize, Expr.constTerm(0.0), binary)
        finally:
            cons.remove()

        if value == "infeasible":
            infeasible_states.append(state)

    if len(infeasible_states) == 1:
        return 1 - infeasible_states[0]

    return -1

def solve_aggregate_integral_bounds(expression, lower, upper, binary, epsilon=1e-4):
    if lower == upper:
        return [lower, upper]

    value = solve_mosek_problem(ObjectiveSense.Minimize, expression, binary)
    if value not in ['unknown', 'infeasible']:
        candidate = int(np.ceil(value - epsilon * max(abs(value), 1)))
        lower = max(lower, candidate)

    value = solve_mosek_problem(ObjectiveSense.Maximize, expression, binary)
    if value not in ['unknown', 'infeasible']:
        candidate = int(np.floor(value + epsilon * max(abs(value), 1)))
        upper = min(upper, candidate)

    return [lower, upper]

def get_bounds(junctions, tanks, pipes, pumps, valves, num_periods):
    nodes, links = junctions | tanks, pipes | pumps | valves

    return [{'flow': {label: [link.flow_lower_bound[t], link.flow_upper_bound[t]] for label, link in links.items()},
             'head': {label: [node.head_lower_bound[t], node.head_upper_bound[t]] for label, node in nodes.items()},
             'next_period_head': {label: [tank.head_lower_bound[t + 1], tank.head_upper_bound[t + 1]] for label, tank in tanks.items()},
             'pump_status': {label: pump.decisions[t] for label, pump in pumps.items()},
             'valve_status': {label: valve.decisions[t] for label, valve in valves.items()}} for t in range(num_periods)]

def adjust_variable_bounds(expression, lower, upper, current_lower, current_upper, label):
    lower, upper = max(lower, current_lower), min(upper, current_upper)
    if lower > current_lower:
        if label + '_lb' in _cached_worker_model._obbt_bound_cons:
            _cached_worker_model._obbt_bound_cons[label + '_lb'].remove()
        _cached_worker_model._obbt_bound_cons[label + '_lb'] = _cached_worker_model.constraint(expression >= lower)
    if upper < current_upper:
        if label + '_ub' in _cached_worker_model._obbt_bound_cons:
            _cached_worker_model._obbt_bound_cons[label + '_ub'].remove()
        _cached_worker_model._obbt_bound_cons[label + '_ub'] = _cached_worker_model.constraint(expression <= upper)
    return [lower, upper]

def adjust_bounds(earlier_bounds):
    for t, bounds in enumerate(earlier_bounds):
        for label, (lb, ub) in bounds['flow'].items():
            link = _cached_worker_model._links[label]
            link_index = _cached_worker_model._link_index[label]
            link.flow_lower_bound[t], link.flow_upper_bound[t] = adjust_variable_bounds(_cached_worker_model._flow[link_index, t], lb, ub, link.flow_lower_bound[t], link.flow_upper_bound[t], f'flow_{link_index}_{t}')

        for label, (lb, ub) in bounds['head'].items():
            node = _cached_worker_model._nodes[label]
            node_index = _cached_worker_model._node_index[label]
            node.head_lower_bound[t], node.head_upper_bound[t] = adjust_variable_bounds(_cached_worker_model._head[node_index, t], lb, ub, node.head_lower_bound[t], node.head_upper_bound[t], f'head_{node_index}_{t}')

        for variable, arcs in [('pump_status', _cached_worker_model._pumps), ('valve_status', _cached_worker_model._valves)]:
            for index, (label, arc) in enumerate(arcs.items()):
                status = bounds[variable][label]
                if status == -1 or arc.decisions[t] == status: continue
                _cached_worker_model.constraint(getattr(_cached_worker_model, '_' + variable)[index, t], Domain.equalsTo(float(status)))
                arc.decisions[t] = status

    for tank_index, label in enumerate(_cached_worker_model._set_of_tanks):
        tank = _cached_worker_model._tanks[label]
        lb, ub = earlier_bounds[-1]['next_period_head'][label]
        tank.head_lower_bound[_cached_worker_model._time_steps], tank.head_upper_bound[_cached_worker_model._time_steps] = adjust_variable_bounds(_cached_worker_model._tank_terminal_head[tank_index], lb, ub, tank.head_lower_bound[_cached_worker_model._time_steps], tank.head_upper_bound[_cached_worker_model._time_steps], f'terminal_head_{tank_index}')

    if _cached_worker_model._set_of_nodes:
        _cached_worker_model._head_lb.setValue(np.stack([_cached_worker_model._nodes[n].head_lower_bound[:_cached_worker_model._time_steps] for n in _cached_worker_model._set_of_nodes]))
        _cached_worker_model._head_ub.setValue(np.stack([_cached_worker_model._nodes[n].head_upper_bound[:_cached_worker_model._time_steps] for n in _cached_worker_model._set_of_nodes]))

    if len(_cached_worker_model._set_of_links) > 0:
        _cached_worker_model._flow_lb.setValue(np.stack([_cached_worker_model._links[l].flow_lower_bound for l in _cached_worker_model._set_of_links]))
        _cached_worker_model._flow_ub.setValue(np.stack([_cached_worker_model._links[l].flow_upper_bound for l in _cached_worker_model._set_of_links]))
        _cached_worker_model._flow_lb_pos.setValue(np.stack([np.maximum(_cached_worker_model._links[l].flow_lower_bound, 0.0) for l in _cached_worker_model._set_of_links]))
        _cached_worker_model._flow_ub_neg.setValue(np.stack([np.minimum(_cached_worker_model._links[l].flow_upper_bound, 0.0) for l in _cached_worker_model._set_of_links]))

def perform_obbt(t, binary, earlier_bounds = None):
    bounds = {'flow': {}, 'head': {}, 'next_period_head': {}, 'pump_status': {}, 'valve_status': {}}

    if earlier_bounds is not None:
        adjust_bounds(earlier_bounds)

        if len(_cached_worker_model._set_of_pipes) > 0:
            for cons in _cached_worker_model._pipe_facet_cons:
                cons.remove()
            find_hydraulic_loss_facets(_cached_worker_model._pipes, _cached_worker_model._time_steps)
            AddPipeFacets(_cached_worker_model)

        if len(_cached_worker_model._set_of_valves) > 0:
            for cons in _cached_worker_model._valve_facet_cons:
                cons.remove()
            find_hydraulic_loss_facets(_cached_worker_model._valves, _cached_worker_model._time_steps)
            AddValveFacets(_cached_worker_model)

        if len(_cached_worker_model._set_of_pumps) > 0:
            for cons in _cached_worker_model._pump_facet_cons:
                cons.remove()
            find_head_gain_facets(_cached_worker_model._pumps, _cached_worker_model._time_steps)
            AddPumpFacets(_cached_worker_model)

    for link in _cached_worker_model._set_of_links:
        bounds['flow'][link] = solve_variable_bounds(_cached_worker_model._flow[_cached_worker_model._link_index[link],t], _cached_worker_model._links[link].flow_lower_bound[t], _cached_worker_model._links[link].flow_upper_bound[t], binary, f'flow_{_cached_worker_model._link_index[link]}_{t}')
        _cached_worker_model._links[link].flow_lower_bound[t], _cached_worker_model._links[link].flow_upper_bound[t] = bounds['flow'][link]

    for node in _cached_worker_model._set_of_nodes:
        bounds['head'][node] = solve_variable_bounds(_cached_worker_model._head[_cached_worker_model._node_index[node],t], _cached_worker_model._nodes[node].head_lower_bound[t], _cached_worker_model._nodes[node].head_upper_bound[t], binary, f'head_{_cached_worker_model._node_index[node]}_{t}')
        _cached_worker_model._nodes[node].head_lower_bound[t], _cached_worker_model._nodes[node].head_upper_bound[t] = bounds['head'][node]

    for tank_index, tank in enumerate(_cached_worker_model._set_of_tanks):
        expression = _cached_worker_model._head[_cached_worker_model._node_index[tank], t + 1] if t + 1 < _cached_worker_model._time_steps else _cached_worker_model._tank_terminal_head[tank_index]
        label = f'head_{_cached_worker_model._node_index[tank]}_{t + 1}' if t + 1 < _cached_worker_model._time_steps else f'terminal_head_{tank_index}'
        bounds['next_period_head'][tank] = solve_variable_bounds(expression, _cached_worker_model._tanks[tank].head_lower_bound[t+1], _cached_worker_model._tanks[tank].head_upper_bound[t+1], binary, label)
        _cached_worker_model._tanks[tank].head_lower_bound[t+1], _cached_worker_model._tanks[tank].head_upper_bound[t+1] = bounds['next_period_head'][tank]

    for variable, arcs in [('pump_status', _cached_worker_model._pumps), ('valve_status', _cached_worker_model._valves)]:
        for index, (label, arc) in enumerate(arcs.items()):
            status = arc.decisions[t]
            if status == -1:
                expression = getattr(_cached_worker_model, '_' + variable)[index, t]
                status = solve_variable_binary(expression, binary)
                if status in (0, 1):
                    _cached_worker_model.constraint(expression, Domain.equalsTo(float(status)))
                    arc.decisions[t] = status
                    if status == 0:
                        bounds['flow'][label] = adjust_variable_bounds(_cached_worker_model._flow[_cached_worker_model._link_index[label], t], 0.0, 0.0, arc.flow_lower_bound[t], arc.flow_upper_bound[t], f'flow_{_cached_worker_model._link_index[label]}_{t}')
                        arc.flow_lower_bound[t], arc.flow_upper_bound[t] = bounds['flow'][label]
            bounds[variable][label] = status

    return bounds

def update_bounds(junctions, tanks, pipes, pumps, valves, results, time_steps):
    shrinkage = {'flow': [], 'head': [], 'pump_status': [], 'valve_status': []}
    nodes, links = junctions | tanks, pipes | pumps | valves

    for label, tank in tanks.items():
        for time_step in range(time_steps + 1):
            width = tank.head_upper_bound[time_step] - tank.head_lower_bound[time_step]
            tank.head_lower_bound[time_step] = max(tank.head_lower_bound[time_step], results[time_step]['head'][label][0] if time_step < time_steps else -np.inf, results[time_step - 1]['next_period_head'][label][0] if time_step > 0 else -np.inf)
            tank.head_upper_bound[time_step] = min(tank.head_upper_bound[time_step], results[time_step]['head'][label][1] if time_step < time_steps else np.inf, results[time_step - 1]['next_period_head'][label][1] if time_step > 0 else np.inf)

            if time_step < time_steps:
                results[time_step]['head'][label] = [tank.head_lower_bound[time_step], tank.head_upper_bound[time_step]]

            if time_step > 0:
                results[time_step - 1]['next_period_head'][label] = [tank.head_lower_bound[time_step], tank.head_upper_bound[time_step]]

            if width > 1e-9:
                shrinkage['head'].append(1 - ((tank.head_upper_bound[time_step] - tank.head_lower_bound[time_step]) / width))

    for time_step, result in results.items():
        for label, (lb, ub) in result['flow'].items():
            link = links[label]
            width = link.flow_upper_bound[time_step] - link.flow_lower_bound[time_step]
            link.flow_lower_bound[time_step], link.flow_upper_bound[time_step] = lb, ub
            result['flow'][label] = [lb, ub]
            if width > 1e-9:
                shrinkage['flow'].append(1 - ((ub - lb) / width))

        for label, (lb, ub) in result['head'].items():
            if label in tanks: continue
            node = nodes[label]
            width = node.head_upper_bound[time_step] - node.head_lower_bound[time_step]
            node.head_lower_bound[time_step], node.head_upper_bound[time_step] = lb, ub
            result['head'][label] = [lb, ub]
            if width > 1e-9:
                shrinkage['head'].append(1 - ((ub - lb) / width))

        for variable, arcs in (("pump_status", pumps), ("valve_status", valves)):
            for label, status in result[variable].items():
                arc = arcs[label]

                if arc.decisions[time_step] != -1 or status not in (0, 1):
                    continue

                arc.decisions[time_step] = status
                shrinkage[variable].append(1)

                if status == 0:
                    arc.flow_lower_bound[time_step] = arc.flow_upper_bound[time_step] = 0.0
                    result["flow"][label] = [0.0, 0.0]

    return 100 * float(np.mean(shrinkage['flow'])) if shrinkage['flow'] else 0.0, 100 * float(np.mean(shrinkage['head'])) if shrinkage['head'] else 0.0, len(shrinkage['pump_status']), len(shrinkage['valve_status'])

def mosek_obbt(junctions, tanks, pipes, pumps, valves, num_periods, hydraulic_timestep, results_directory, horizons, time_limit, num_rounds=20, binary_pipe=False, binary_pump_valve=False, sheet_prefix="obbt", log_mode="w"):
    write_results(results_directory, "obbt_results.txt", f"\n=== OBBT: {sheet_prefix} ===\nhorizons: {horizons}", mode=log_mode)
    binary = (binary_pipe and bool(pipes)) or (binary_pump_valve and bool(pumps or valves))
    executors = {}

    try:
        for period, horizon in enumerate(horizons):
            executors[period] = concurrent.futures.ProcessPoolExecutor(
                max_workers=1,
                initializer=init_obbt,
                initargs=(f"obbt_{period}", results_directory, junctions, tanks, pipes, pumps, valves, horizon, hydraulic_timestep, time_limit, binary_pipe, binary_pump_valve)
            )

        for r in range(num_rounds):
            round_start = time.time()
            earlier_bounds = get_bounds(junctions, tanks, pipes, pumps, valves, num_periods) if r else None

            futures = {}
            for period, (start, end) in enumerate(horizons):
                futures[period] = executors[period].submit(perform_obbt, period - start, binary, earlier_bounds[start:end] if r else None)

            results = {period: future.result() for period, future in futures.items()}
            flow_shrinkage, head_shrinkage, pump_statuses_fixed, valve_statuses_fixed = update_bounds(junctions, tanks, pipes, pumps, valves, results, num_periods)
            write_results(results_directory, 'obbt_results.txt', f'Round {r+1} time : {time.time() - round_start}')
            write_results(results_directory, 'obbt_results.txt', f"flow : {flow_shrinkage:.2f}% | head : {head_shrinkage:.2f}% | pump_status fixed : {pump_statuses_fixed} | valve_status fixed : {valve_statuses_fixed}")
            write_results(results_directory, 'obbt_results.txt', '-------------------------------------------')
            write_bounds(junctions, tanks, pipes, pumps, valves, num_periods, results_directory, sheet_name=f'{sheet_prefix}_{r}', output_file='decision_variable_bounds.xlsx')

            if flow_shrinkage < 1.0 and head_shrinkage < 1.0 and pump_statuses_fixed + valve_statuses_fixed == 0:
                break

    finally:
        for executor in executors.values():
            executor.shutdown(wait=True)

def generate_aggregate_inequalities(binary, variable_subset, horizon, epsilon):
    inequalities = set()
    expressions = []
    terms = []

    start, end = horizon

    for variable, component in variable_subset:
        labels = getattr(_cached_worker_model, '_set_of_' + component)

        if not labels:
            continue

        decision_variable = getattr(_cached_worker_model, '_' + variable)
        expressions.append(Expr.sum(decision_variable[:, start:end]))
        terms.extend((variable, label, t, 1.0) for label in labels for t in range(start, end))

    if not terms:
        return inequalities

    expression = Expr.add(expressions)

    lower = 0
    upper = len(terms)

    lb, ub = solve_aggregate_integral_bounds(expression, lower, upper, binary, epsilon)

    if lb > lower:
        inequalities.add(valid_inequality_key(terms, 'ge', lb))

    if ub < upper:
        inequalities.add(valid_inequality_key(terms, 'le', ub))

    return inequalities

def mosek_aggregate_inequalities(junctions, tanks, pipes, pumps, valves, num_periods, hydraulic_timestep, results_directory, variable_subsets, horizons, time_limit, binary_pipe = False, binary_pump_valve = False, epsilon=1e-4, num_workers = 4):
    start_time = time.time()
    inequalities = set()

    jobs = [(variable_subset, horizon) for variable_subset in variable_subsets for horizon in horizons]
    binary = (binary_pipe and bool(pipes)) or (binary_pump_valve and bool(pumps or valves))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(num_workers, len(jobs)),
        initializer=init_obbt,
        initargs=('aggregate_inequalities', results_directory, junctions, tanks, pipes, pumps, valves, (0, num_periods), hydraulic_timestep, time_limit, binary_pipe, binary_pump_valve)
    ) as executor:

        futures = [executor.submit(generate_aggregate_inequalities, binary, variable_subset, horizon, epsilon) for variable_subset, horizon in jobs]

        for future in futures:
            inequalities.update(future.result())

    write_results(results_directory, 'valid_inequalities_results.txt', f'aggregate | jobs : {len(jobs)} | inequalities : {len(inequalities)} | time : {time.time() - start_time}')

    return inequalities