import os
import time
import numpy as np
from copy import deepcopy
from gurobipy import GRB
from .helpers import write_results
from .read_epanet import set_timesteps
from .operational.ipopt_iterative import IpoptFixedOperationalModel
from .operational.mosek import MosekOperationalModel, SolveMosekOperationalModel, AddNoGoodCut
from .operational.gurobi_soc import GurobiOperationalModelSoc, SolveGurobiOperationalModelSoc, AddNoGoodCutSoc

def run_iterative_ipopt(current_solution, period_objects, tanks, results_directory, time_steps, hydraulic_timestep):
    up_to = time_steps
    ipopt_solution = {}
    for time_step in range(time_steps):
        period_junctions, period_tanks, period_pipes, period_pumps, period_valves = period_objects[time_step]
        period_fixed_solution = {'flow': {link: current_solution['flow'][link][time_step] for link in period_pipes | period_pumps | period_valves},
                                 'pump_status': {pump: current_solution['pump_status'][pump][time_step] for pump in period_pumps},
                                 'valve_status': {valve: current_solution['valve_status'][valve][time_step] for valve in period_valves}}

        if time_step > 0:
            for label, tank in period_tanks.items():
                tank.head_lower_bound[0] = tank.head_upper_bound[0] = period_ipopt_solution['next_head'][label]
        period_ipopt_status, period_ipopt_solution = IpoptFixedOperationalModel(f'ipopt fixed iterative {time_step}', results_directory, period_junctions, period_tanks, period_pipes, period_pumps, period_valves, hydraulic_timestep, period_fixed_solution)

        if period_ipopt_status == 'feasible':
            period_nodes = period_junctions | period_tanks
            for node, head in period_ipopt_solution['head'].items():
                if node in period_tanks.keys() or period_nodes[node].segment == 'reservoir': continue
                if head < period_nodes[node].head_lower_bound[0] - 1e-3 or head > period_nodes[node].head_upper_bound[0] + 1e-3:
                    period_ipopt_status = 'infeasible'
                    break
            for tank, head in period_ipopt_solution['next_head'].items():
                if head < tanks[tank].head_lower_bound[time_step+1] - 1e-3 or head > tanks[tank].head_upper_bound[time_step+1] + 1e-3:
                    period_ipopt_status = 'infeasible'
                    break
            period_links = period_pipes | period_pumps | period_valves
            for link, flow in period_ipopt_solution['flow'].items():
                if flow < period_links[link].flow_lower_bound[0] - 1e-3 or flow > period_links[link].flow_upper_bound[0] + 1e-3:
                    period_ipopt_status = 'infeasible'
                    break

        if period_ipopt_status != 'feasible':
            up_to = time_step + 1
            break

        ipopt_solution[time_step] = period_ipopt_solution

    return ipopt_solution, period_ipopt_status, up_to

def operational_problem_two_phased(results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit):
    period_objects = {}
    for time_step in range(time_steps):
        period_junctions, period_tanks, period_pipes, period_pumps, period_valves = deepcopy((junctions, tanks, pipes, pumps, valves))
        period_objects[time_step] = set_timesteps(period_junctions, period_tanks, period_pipes, period_pumps, period_valves, [time_step])

    mosek_model = MosekOperationalModel('two_phased mosek', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit)
    mosek_model.setSolverParam("mioMaxNumSolutions", 1)

    lower_bound, upper_bound, incumbent, status = None, None, None, 'nosolution'
    incumbent_count, hydraulic_feasible_count = 0, 0
    start_time = time.time()

    while True:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0: break
        mosek_model.setSolverParam('mioMaxTime', remaining_time)

        mosek_status, mosek_solution, mosek_lower_bound = SolveMosekOperationalModel(mosek_model, 'two_phased mosek', results_directory)
        if mosek_lower_bound is not None and (lower_bound is None or mosek_lower_bound > lower_bound): lower_bound = mosek_lower_bound

        if mosek_status != 'feasible' or mosek_solution is None:
            break

        incumbent_count += 1
        ipopt_solution, ipopt_status, up_to = run_iterative_ipopt(mosek_solution, period_objects, tanks, results_directory, time_steps, hydraulic_timestep)
        AddNoGoodCut(mosek_model, mosek_solution, up_to=up_to)

        if ipopt_status == 'feasible':
            hydraulic_feasible_count += 1
            candidate_upper_bound = sum(pumps[pump].fixed_cost[t] * mosek_solution['pump_status'][pump][t] + pumps[pump].operational_cost[t] * ipopt_solution[t]['flow'].get(pump, 0.0) for t in range(time_steps) for pump in pumps)
            if upper_bound is None or candidate_upper_bound < upper_bound:
                upper_bound, incumbent, status = candidate_upper_bound, ipopt_solution, 'feasible'
                mosek_model.setSolverParam("upperObjCut", candidate_upper_bound)

    if lower_bound is not None and upper_bound is not None:
        lower_bound = min(lower_bound, upper_bound)

    mosek_model.dispose()
    return status, incumbent, lower_bound, upper_bound, time.time() - start_time, incumbent_count, hydraulic_feasible_count

def operational_problem_two_phased_soc(results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit, valid_inequalities = None):
    period_objects = {}
    for time_step in range(time_steps):
        period_junctions, period_tanks, period_pipes, period_pumps, period_valves = deepcopy((junctions, tanks, pipes, pumps, valves))
        period_objects[time_step] = set_timesteps(period_junctions, period_tanks, period_pipes, period_pumps, period_valves, [time_step])

    gurobi_model, gurobi_model_env = GurobiOperationalModelSoc('two_phased gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, valid_inequalities=valid_inequalities)
    gurobi_model.setParam('SolutionLimit', 1)

    lower_bound, upper_bound, incumbent, status = None, None, None, 'nosolution'
    incumbent_count, hydraulic_feasible_count = 0, 0
    start_time = time.time()

    while True:
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0: break
        gurobi_model.setParam('TimeLimit', remaining_time)

        gurobi_status, gurobi_solution, gurobi_lower_bound = SolveGurobiOperationalModelSoc(gurobi_model, 'two_phased gurobi', results_directory)
        if gurobi_lower_bound is not None and (lower_bound is None or gurobi_lower_bound > lower_bound): lower_bound = gurobi_lower_bound

        if gurobi_status != 'feasible' or gurobi_solution is None:
            break

        incumbent_count += 1
        ipopt_solution, ipopt_status, up_to = run_iterative_ipopt(gurobi_solution, period_objects, tanks, results_directory, time_steps, hydraulic_timestep)
        AddNoGoodCutSoc(gurobi_model, gurobi_solution, up_to=up_to)

        if ipopt_status == 'feasible':
            hydraulic_feasible_count += 1
            candidate_upper_bound = sum(pumps[pump].fixed_cost[t] * gurobi_solution['pump_status'][pump][t] + pumps[pump].operational_cost[t] * ipopt_solution[t]['flow'].get(pump, 0.0) for t in range(time_steps) for pump in pumps)
            if upper_bound is None or candidate_upper_bound < upper_bound:
                upper_bound, incumbent, status = candidate_upper_bound, ipopt_solution, 'feasible'
                gurobi_model.setParam('Cutoff', candidate_upper_bound)

    if lower_bound is not None and upper_bound is not None:
        lower_bound = min(lower_bound, upper_bound)

    gurobi_model.dispose()
    gurobi_model_env.dispose()
    return status, incumbent, lower_bound, upper_bound, time.time() - start_time, incumbent_count, hydraulic_feasible_count

def gurobi_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        callback_data = model._callback_data
        current_solution = {}
        binary_assignment = ()

        if len(model._set_of_pumps) > 0:
            pump_values = model.cbGetSolution(model._pump_status)
            current_solution['pump_status'] = {pump: np.rint([pump_values[pump, t] for t in model._set_of_time_periods]) for pump in model._set_of_pumps}
            binary_assignment = binary_assignment + tuple(int(value) for pump in model._set_of_pumps for value in current_solution['pump_status'][pump])

        if len(model._set_of_valves) > 0:
            valve_values = model.cbGetSolution(model._valve_status)
            current_solution['valve_status'] = {valve: np.rint([valve_values[valve, t] for t in model._set_of_time_periods]) for valve in model._set_of_valves}
            binary_assignment = binary_assignment + tuple(int(value) for valve in model._set_of_valves for value in current_solution['valve_status'][valve])

        if binary_assignment in callback_data['cut_assignments'].keys():
            AddNoGoodCutSoc(model, current_solution, lazy_flag=True, up_to=callback_data['cut_assignments'][binary_assignment])
            return

        if len(model._set_of_links) > 0:
            flow_values = model.cbGetSolution(model._flow)
            current_solution['flow'] = {link: np.array([flow_values[link, t] for t in model._set_of_time_periods]) for link in model._set_of_links}

        callback_data['incumbent_count'] += 1
        ipopt_solution, ipopt_status, up_to = run_iterative_ipopt(current_solution, callback_data['period_objects'], callback_data['tanks'], callback_data['results_directory'], callback_data['time_steps'], callback_data['hydraulic_timestep'])
        callback_data['cut_assignments'][binary_assignment] = up_to
        AddNoGoodCutSoc(model, current_solution, lazy_flag=True, up_to=up_to)

        if ipopt_status == 'feasible':
            callback_data['hydraulic_feasible_count'] += 1
            candidate_upper_bound = sum(callback_data['pumps'][pump].fixed_cost[t] * current_solution['pump_status'][pump][t] + callback_data['pumps'][pump].operational_cost[t] * ipopt_solution[t]['flow'].get(pump, 0.0) for t in range(callback_data['time_steps']) for pump in callback_data['pumps'])
            if callback_data['upper_bound'] is None or candidate_upper_bound < callback_data['upper_bound']:
                callback_data['upper_bound'], callback_data['incumbent'], callback_data['status'] = candidate_upper_bound, ipopt_solution, 'feasible'

def operational_problem_soc_callback(results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit, valid_inequalities):
    period_objects = {}
    for time_step in range(time_steps):
        period_junctions, period_tanks, period_pipes, period_pumps, period_valves = deepcopy((junctions, tanks, pipes, pumps, valves))
        period_objects[time_step] = set_timesteps(period_junctions, period_tanks, period_pipes, period_pumps, period_valves, [time_step])

    gurobi_model, gurobi_model_env = GurobiOperationalModelSoc('callback gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, valid_inequalities=valid_inequalities)
    gurobi_model.setParam('TimeLimit', time_limit)
    gurobi_model.setParam('LazyConstraints', 1)
    #gurobi_model.setParam('MIPFocus', 1)
    #gurobi_model.setParam('PreCrush', 1)
    #gurobi_model.setParam('MIPGap', 0.0)

    gurobi_model._callback_data = {'results_directory': results_directory, 'period_objects': period_objects, 'time_steps': time_steps, 'tanks': tanks, 
                                   'pumps': pumps, 'hydraulic_timestep': hydraulic_timestep, 'status': 'nosolution', 'upper_bound': None, 
                                   'incumbent': None, 'cut_assignments': {}, 'incumbent_count': 0, 'hydraulic_feasible_count': 0}

    start_time = time.time()
    gurobi_model.optimize(gurobi_callback)
    callback_data = gurobi_model._callback_data
    lower_bound = gurobi_model.ObjBound if gurobi_model.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and abs(gurobi_model.ObjBound) < GRB.INFINITY else None
    if lower_bound is not None and callback_data['upper_bound'] is not None:
        lower_bound = min(lower_bound, callback_data['upper_bound'])

    gurobi_model.dispose()
    gurobi_model_env.dispose()
    return callback_data['status'], callback_data['incumbent'], lower_bound, callback_data['upper_bound'], time.time() - start_time, callback_data['incumbent_count'], callback_data['hydraulic_feasible_count']