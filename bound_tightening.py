import os
import time
import numpy as np
import concurrent.futures
from mosek.fusion import *
import mosek.fusion.pythonic
from read_epanet import set_timesteps
from hull_facets import find_pipe_facets, find_pump_facets
from helpers import round_bound, write_bounds, write_results
from operational_mosek import MosekOperationalModel, AddPipeFacets, AddPumpFacets

_cached_worker_model = None

def init_obbt_task(model_name, results_directory, junctions, tanks, pipes, pumps, valves, time_step, hydraulic_timestep, head_loss_equation):
    global _cached_worker_model
    set_timesteps(junctions, tanks, pipes, pumps, valves, [time_step])
    if len(pipes) > 0:
        find_pipe_facets(pipes, 1, head_loss_equation)
    if len(pumps) > 0:
        find_pump_facets(pumps, 1)
    _cached_worker_model = MosekOperationalModel(model_name, results_directory, junctions, tanks, pipes, pumps, valves, 1, hydraulic_timestep, head_loss_equation, False, False, True)
    _cached_worker_model.setSolverParam("numThreads", 1)
    _cached_worker_model.setSolverParam('intpntSolveForm', 'primal')
    _cached_worker_model._obbt_bound_cons = {}

def solve_mosek_problem(sense, expression, binary=False):
    _cached_worker_model.objective('objective', sense, expression)
    try:
        _cached_worker_model.solve()
    except (OptimizeError, SolutionError):
        return "unknown"

    if _cached_worker_model.getProblemStatus() in (ProblemStatus.PrimalInfeasible, ProblemStatus.PrimalAndDualInfeasible): return "infeasible"

    try:
        if binary:
            bound = _cached_worker_model.getSolverDoubleInfo("mioObjBound")
        else:
            dual_status = _cached_worker_model.getDualSolutionStatus()
            if dual_status not in (SolutionStatus.Optimal, SolutionStatus.Feasible): return "unknown"
            _cached_worker_model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
            bound = _cached_worker_model.dualObjValue()

        return bound if np.isfinite(bound) else "unknown"

    except SolutionError:
        return "unknown"

def solve_variable_bounds(expression, lower, upper, label=None, epsilon=1e-3):
    if upper - lower <= epsilon:
        return [lower, upper]

    value = solve_mosek_problem(ObjectiveSense.Minimize, expression)
    if value not in ['unknown', 'infeasible']:
        candidate = round_bound(value - epsilon * max(abs(value), 1), 'floor')
        if candidate > lower:
            lower = candidate
            if label is not None and label + '_lb' in _cached_worker_model._obbt_bound_cons:
                _cached_worker_model._obbt_bound_cons[label + '_lb'].remove()
            if label is not None:
                cons = _cached_worker_model.constraint(expression >= lower)
                _cached_worker_model._obbt_bound_cons[label + '_lb'] = cons

    value = solve_mosek_problem(ObjectiveSense.Maximize, expression)
    if value not in ['unknown', 'infeasible']:
        candidate = round_bound(value + epsilon * max(abs(value), 1), 'ceiling')
        if candidate < upper:
            upper = candidate
            if label is not None and label + '_ub' in _cached_worker_model._obbt_bound_cons:
                _cached_worker_model._obbt_bound_cons[label + '_ub'].remove()
            if label is not None:
                cons = _cached_worker_model.constraint(expression <= upper)
                _cached_worker_model._obbt_bound_cons[label + '_ub'] = cons

    return [lower, upper]

def solve_variable_binary(expression):
    cons = _cached_worker_model.constraint(expression, Domain.equalsTo(1.0))
    value = solve_mosek_problem(ObjectiveSense.Minimize, Expr.constTerm(0.0))
    cons.remove()
    if value == 'infeasible': return 0

    cons = _cached_worker_model.constraint(expression, Domain.equalsTo(0.0))
    value = solve_mosek_problem(ObjectiveSense.Minimize, Expr.constTerm(0.0))
    cons.remove()
    if value == 'infeasible': return 1

    return -1

def adjust_bounds(_cached_worker_model, earlier_bounds, current_storage, ahead_storage):
    for pump, status in earlier_bounds['pump_status'].items():
        _cached_worker_model._pumps[pump].decisions[0] = 1 if status == 1 else 0 if status == 0 else -1

    for valve, status in earlier_bounds['valve_status'].items():
        _cached_worker_model._valves[valve].decisions[0] = 1 if status == 1 else 0 if status == 0 else -1

    for link, (lb, ub) in earlier_bounds['flow'].items():
        _cached_worker_model._links[link].flow_lower_bound[0] = lb
        _cached_worker_model._links[link].flow_upper_bound[0] = ub

    for node, (lb, ub) in earlier_bounds['head'].items():
        _cached_worker_model._nodes[node].head_lower_bound[0] = lb
        _cached_worker_model._nodes[node].head_upper_bound[0] = ub

    for i, tank in enumerate(_cached_worker_model._set_of_tanks):
        _cached_worker_model._tanks[tank].storage_lower_bound[0] = current_storage[tank][0]
        _cached_worker_model._tanks[tank].storage_upper_bound[0] = current_storage[tank][1]
        _cached_worker_model._tanks[tank].storage_lower_bound[1] = ahead_storage[tank][0]
        _cached_worker_model._tanks[tank].storage_upper_bound[1] = ahead_storage[tank][1]

        storage_expressions = {f'external_storage_{i}_0_lb': _cached_worker_model._storage[i, 0] >= current_storage[tank][0],
                               f'external_storage_{i}_0_ub': _cached_worker_model._storage[i, 0] <= current_storage[tank][1],
                               f'external_storage_{i}_1_lb': _cached_worker_model._storage[i, 1] >= ahead_storage[tank][0],
                               f'external_storage_{i}_1_ub': _cached_worker_model._storage[i, 1] <= ahead_storage[tank][1]}

        for key in storage_expressions:
            if key in _cached_worker_model._obbt_bound_cons:
                _cached_worker_model._obbt_bound_cons[key].remove()

        for key, expression in storage_expressions.items():
            _cached_worker_model._obbt_bound_cons[key] = _cached_worker_model.constraint(expression)

    if len(_cached_worker_model._set_of_nodes) > 0:
        _cached_worker_model._head_lb.setValue(np.stack([_cached_worker_model._nodes[n].head_lower_bound for n in _cached_worker_model._set_of_nodes]))
        _cached_worker_model._head_ub.setValue(np.stack([_cached_worker_model._nodes[n].head_upper_bound for n in _cached_worker_model._set_of_nodes]))

    if len(_cached_worker_model._set_of_links) > 0:
        _cached_worker_model._flow_lb.setValue(np.stack([_cached_worker_model._links[l].flow_lower_bound for l in _cached_worker_model._set_of_links]))
        _cached_worker_model._flow_ub.setValue(np.stack([_cached_worker_model._links[l].flow_upper_bound for l in _cached_worker_model._set_of_links]))

        _cached_worker_model._flow_lb_pos.setValue(np.stack([np.maximum(_cached_worker_model._links[l].flow_lower_bound, 0.0) for l in _cached_worker_model._set_of_links]))
        _cached_worker_model._flow_ub_neg.setValue(np.stack([np.minimum(_cached_worker_model._links[l].flow_upper_bound, 0.0) for l in _cached_worker_model._set_of_links]))

def perform_obbt(head_loss_equation, earlier_bounds = None, current_storage = None, ahead_storage = None):
    bounds = {'flow': {}, 'head': {}, 'storage': {}, 'pump_status': {}, 'valve_status': {}}

    if earlier_bounds is not None:
        adjust_bounds(_cached_worker_model, earlier_bounds, current_storage, ahead_storage)

        if len(_cached_worker_model._set_of_pipes) > 0:
            for cons in _cached_worker_model._pipe_facet_cons:
                cons.remove()
            find_pipe_facets(_cached_worker_model._pipes, 1, head_loss_equation)
            AddPipeFacets(_cached_worker_model)

        if len(_cached_worker_model._set_of_pumps) > 0:
            for cons in _cached_worker_model._pump_facet_cons:
                cons.remove()
            find_pump_facets(_cached_worker_model._pumps, 1)
            AddPumpFacets(_cached_worker_model)

    for link in _cached_worker_model._set_of_links:
        bounds['flow'][link] = solve_variable_bounds(_cached_worker_model._flow[_cached_worker_model._link_index[link],0], _cached_worker_model._links[link].flow_lower_bound[0], _cached_worker_model._links[link].flow_upper_bound[0], f'flow_{_cached_worker_model._link_index[link]}_0')

    for node in _cached_worker_model._set_of_nodes:
        bounds['head'][node] = solve_variable_bounds(_cached_worker_model._head[_cached_worker_model._node_index[node],0], _cached_worker_model._nodes[node].head_lower_bound[0], _cached_worker_model._nodes[node].head_upper_bound[0], f'head_{_cached_worker_model._node_index[node]}_0')

    for i, tank in enumerate(_cached_worker_model._tanks.values()):
        bounds['storage'][tank.label] = solve_variable_bounds(_cached_worker_model._storage[i,1], tank.storage_lower_bound[1], tank.storage_upper_bound[1], f'storage_{i}_1')

    for pump_index, pump in enumerate(_cached_worker_model._set_of_pumps):
        if _cached_worker_model._pumps[pump].decisions[0] != -1: continue
        status = solve_variable_binary(_cached_worker_model._pump_status[pump_index,0])
        if status == 1:
            _cached_worker_model.constraint(_cached_worker_model._pump_status[pump_index,0], Domain.equalsTo(1.0))
        elif status == 0:
            _cached_worker_model.constraint(_cached_worker_model._pump_status[pump_index,0], Domain.equalsTo(0.0))
        bounds['pump_status'][pump] = status

    for valve_index, valve in enumerate(_cached_worker_model._set_of_valves):
        if _cached_worker_model._valves[valve].decisions[0] != -1: continue
        status = solve_variable_binary(_cached_worker_model._valve_status[valve_index,0])
        if status == 1:
            _cached_worker_model.constraint(_cached_worker_model._valve_status[valve_index,0], Domain.equalsTo(1.0))
        elif status == 0:
            _cached_worker_model.constraint(_cached_worker_model._valve_status[valve_index,0], Domain.equalsTo(0.0))
        bounds['valve_status'][valve] = status

    return bounds

def update_bounds(junctions, tanks, pipes, pumps, valves, results):
    shrinkage = {'flow': [], 'head': [], 'storage': [], 'pump_status': [], 'valve_status': []}
    nodes, links = junctions | tanks, pipes | pumps | valves

    for time_step, result in results.items():
        for label, (lb, ub) in result['flow'].items():
            link = links[label]
            width = link.flow_upper_bound[time_step] - link.flow_lower_bound[time_step]
            link.flow_lower_bound[time_step], link.flow_upper_bound[time_step] = max(link.flow_lower_bound[time_step], lb), min(link.flow_upper_bound[time_step], ub)
            if width > 1e-9:
                shrinkage['flow'].append(1 - ((link.flow_upper_bound[time_step] - link.flow_lower_bound[time_step]) / width))

        for label, (lb, ub) in result['head'].items():
            node = nodes[label]
            width = node.head_upper_bound[time_step] - node.head_lower_bound[time_step]
            node.head_lower_bound[time_step], node.head_upper_bound[time_step] = max(node.head_lower_bound[time_step], lb), min(node.head_upper_bound[time_step], ub)
            if width > 1e-9:
                shrinkage['head'].append(1 - ((node.head_upper_bound[time_step] - node.head_lower_bound[time_step]) / width))

        for label, (lb, ub) in result['storage'].items():
            tank = tanks[label]
            width = tank.storage_upper_bound[time_step+1] - tank.storage_lower_bound[time_step+1]
            tank.storage_lower_bound[time_step+1], tank.storage_upper_bound[time_step+1] = max(tank.storage_lower_bound[time_step+1], lb), min(tank.storage_upper_bound[time_step+1], ub)
            if width > 1e-9:
                shrinkage['storage'].append(1 - ((tank.storage_upper_bound[time_step+1] - tank.storage_lower_bound[time_step+1]) / width))

        for label, status in result['pump_status'].items():
            if pumps[label].decisions[time_step] != -1: continue
            if status == 1:
                pumps[label].decisions[time_step] = 1
                shrinkage['pump_status'].append(1)
            elif status == 0:
                pumps[label].decisions[time_step] = 0
                pumps[label].flow_lower_bound[time_step], pumps[label].flow_upper_bound[time_step] = 0.0, 0.0
                shrinkage['pump_status'].append(1)
                result['flow'][label] = [0.0, 0.0]

        for label, status in result['valve_status'].items():
            if valves[label].decisions[time_step] != -1: continue
            if status == 1:
                valves[label].decisions[time_step] = 1
                shrinkage['valve_status'].append(1)
            elif status == 0:
                valves[label].decisions[time_step] = 0
                valves[label].flow_lower_bound[time_step], valves[label].flow_upper_bound[time_step] = 0.0, 0.0
                shrinkage['valve_status'].append(1)
                result['flow'][label] = [0.0, 0.0]

    for label, tank in tanks.items():
        for time_step in range(len(tank.head_lower_bound)):
            width = tank.storage_upper_bound[time_step] - tank.storage_lower_bound[time_step]

            tank.storage_lower_bound[time_step] = max(tank.storage_lower_bound[time_step], round_bound((tank.head_lower_bound[time_step] - tank.elevation) * tank.area, 'floor'))
            tank.storage_upper_bound[time_step] = min(tank.storage_upper_bound[time_step], round_bound((tank.head_upper_bound[time_step] - tank.elevation) * tank.area, 'ceiling'))

            tank.head_lower_bound[time_step] = max(tank.head_lower_bound[time_step], round_bound(tank.elevation + (tank.storage_lower_bound[time_step] / tank.area), 'floor'))
            tank.head_upper_bound[time_step] = min(tank.head_upper_bound[time_step], round_bound(tank.elevation + (tank.storage_upper_bound[time_step] / tank.area), 'ceiling'))

            if width > 1e-9:
                shrinkage['storage'].append(1 - ((tank.storage_upper_bound[time_step] - tank.storage_lower_bound[time_step]) / width))

            results[time_step]['head'][label] = [tank.head_lower_bound[time_step], tank.head_upper_bound[time_step]]

            if time_step > 0:
                results[time_step - 1]['storage'][label] = [tank.storage_lower_bound[time_step], tank.storage_upper_bound[time_step]]

    return 100 * float(np.mean(shrinkage['flow'])) if shrinkage['flow'] else 0.0, 100 * float(np.mean(shrinkage['head'])) if shrinkage['head'] else 0.0, 100 * float(np.mean(shrinkage['storage'])) if shrinkage['storage'] else 0.0, len(shrinkage['pump_status']), len(shrinkage['valve_status'])

def mosek_obbt(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, results_directory, num_rounds = 15):
    write_results(results_directory, 'obbt_results.txt', f'OBBT | {time_steps} time steps', mode = 'w')

    executors = {}
    for time_step in range(time_steps):
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            initializer=init_obbt_task,
            initargs=(f'obbt_{time_step}', results_directory, junctions, tanks, pipes, pumps, valves, time_step, hydraulic_timestep, head_loss_equation)
        )
        executors[time_step] = executor

    for r in range(num_rounds):
        round_start = time.time()
        futures = {time_step: executors[time_step].submit(perform_obbt, head_loss_equation, results[time_step] if r else None, current_storage[time_step] if r else None, ahead_storage[time_step] if r else None) for time_step in range(time_steps)}
        results = {time_step: f.result() for time_step, f in futures.items()}
        flow_shrinkage, head_shrinkage, storage_shrinkage, pump_statuses_fixed, valve_statuses_fixed = update_bounds(junctions, tanks, pipes, pumps, valves, results)

        current_storage = {time_step: {label: [tank.storage_lower_bound[time_step], tank.storage_upper_bound[time_step]] for label, tank in tanks.items()} for time_step in range(time_steps)}
        ahead_storage = {time_step: {label: [tank.storage_lower_bound[time_step + 1], tank.storage_upper_bound[time_step + 1]] for label, tank in tanks.items()} for time_step in range(time_steps)}

        write_results(results_directory, 'obbt_results.txt', f'Round {r+1} time : {time.time() - round_start}')
        write_results(results_directory, 'obbt_results.txt', f"flow : {flow_shrinkage:.2f}% | head : {head_shrinkage:.2f}% | storage : {storage_shrinkage:.2f}% | pump_status fixed : {pump_statuses_fixed} | valve_status fixed : {valve_statuses_fixed}")
        write_results(results_directory, 'obbt_results.txt', f'-------------------------------------------')
        write_bounds(junctions, tanks, pipes, pumps, valves, time_steps, results_directory, sheet_name = f'bounds_obbt_{r}', output_file = 'decision_variable_bounds.xlsx')

        if flow_shrinkage < 1.0 and head_shrinkage < 1.0 and storage_shrinkage < 1.0 and pump_statuses_fixed + valve_statuses_fixed == 0:
            break

    for executor in executors.values():
        executor.shutdown(wait=True)