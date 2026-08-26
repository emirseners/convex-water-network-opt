import os
import time
import numpy as np
from copy import deepcopy
from gurobipy import GRB
from .read_epanet import set_timesteps
from .helpers import write_results, write_solution
from .hull_facets import find_pipe_facets, find_pump_facets
from .soc_outer import pipe_soc_outer_app, pump_soc_outer_app
from .operational.pyomo import PyomoOperationalModel
from .operational.gurobi import GurobiOperationalModel
from .operational.ipopt_iterative import IpoptFixedOperationalModel
from .operational.mosek import MosekOperationalModel, SolveMosekOperationalModel, AddNoGoodCut
from .operational.gurobi_soc import GurobiOperationalModelSoc, SolveGurobiOperationalModelSoc, AddNoGoodCutSoc

def operational_problem_two_phased(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, flow_direction = False, num_rounds = 5):
    hull_facet_start = time.time()
    find_pipe_facets(pipes, time_steps, head_loss_equation)
    find_pump_facets(pumps, time_steps)
    hull_facet_time = time.time() - hull_facet_start

    mosek_model = MosekOperationalModel('two_phased mosek', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)

    best_lower_bound = None
    start_time = time.time()
    for r in range(num_rounds):
        mosek_status, mosek_solution, mosek_lower_bound, mosek_upper_bound, mosek_solution_time = SolveMosekOperationalModel(mosek_model, 'two_phased mosek', results_directory)
        write_results(os.path.dirname(results_directory), 'results.txt', f'{epanet_model} | two_phased mosek | {mosek_status} | {f"{mosek_lower_bound:.2f}" if mosek_lower_bound is not None else "-"} | {f"{mosek_upper_bound:.2f}" if mosek_upper_bound is not None else "-"} | {mosek_solution_time:.2f} | - | {f"{hull_facet_time:.2f}" if r == 0 else "-"}')
        if mosek_lower_bound is not None and (best_lower_bound is None or mosek_lower_bound > best_lower_bound): best_lower_bound = mosek_lower_bound

        if mosek_status != 'feasible':
            mosek_model.dispose()
            return mosek_status, None, best_lower_bound, None, time.time() - start_time

        ipopt_status, ipopt_solution, ipopt_lower_bound, ipopt_upper_bound, ipopt_solution_time = PyomoOperationalModel('two_phased', 'two_phased mosek to ipopt epsilon', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, mosek_solution, flow_direction)
        write_results(os.path.dirname(results_directory), 'results.txt', f'{epanet_model} | two_phased mosek to ipopt | {ipopt_status} | {f"{ipopt_lower_bound:.2f}" if ipopt_lower_bound is not None else "-"} | {f"{ipopt_upper_bound:.2f}" if ipopt_upper_bound is not None else "-"} | {ipopt_solution_time:.2f} | - | -')

        if len(pumps) + len(valves) == 0 or ipopt_status == 'feasible': break

        AddNoGoodCut(mosek_model, mosek_solution)

    #if ipopt_status == 'feasible':
        #GurobiOperationalModel('two_phased ipopt check', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, ipopt_solution)

    mosek_model.dispose()
    return ipopt_status, ipopt_solution, best_lower_bound, ipopt_upper_bound, time.time() - start_time

def operational_problem_two_phased_soc(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, flow_direction = False, num_rounds = 2):
    hull_facet_start = time.time()
    find_pipe_facets(pipes, time_steps, head_loss_equation)
    find_pump_facets(pumps, time_steps)
    hull_facet_time = time.time() - hull_facet_start

    pipe_soc_outer_app(pipes, time_steps, head_loss_equation)
    pump_soc_outer_app(pumps, time_steps)

    gurobi_model, gurobi_model_env = GurobiOperationalModelSoc('two_phased gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)

    best_lower_bound = None
    start_time = time.time()
    for r in range(num_rounds):
        gurobi_status, gurobi_solution, gurobi_lower_bound, gurobi_upper_bound, gurobi_solution_time = SolveGurobiOperationalModelSoc(gurobi_model, 'two_phased gurobi', results_directory)
        write_results(os.path.dirname(results_directory), 'results.txt', f'{epanet_model} | two_phased gurobi | {gurobi_status} | {f"{gurobi_lower_bound:.2f}" if gurobi_lower_bound is not None else "-"} | {f"{gurobi_upper_bound:.2f}" if gurobi_upper_bound is not None else "-"} | {gurobi_solution_time:.2f} | - | {f"{hull_facet_time:.2f}" if r == 0 else "-"}')
        if gurobi_lower_bound is not None and (best_lower_bound is None or gurobi_lower_bound > best_lower_bound): best_lower_bound = gurobi_lower_bound

        if gurobi_status != 'feasible':
            gurobi_model.dispose()
            gurobi_model_env.dispose()
            return gurobi_status, None, best_lower_bound, None, time.time() - start_time

        ipopt_status, ipopt_solution, ipopt_lower_bound, ipopt_upper_bound, ipopt_solution_time = PyomoOperationalModel('two_phased', 'two_phased gurobi to ipopt epsilon', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, gurobi_solution, flow_direction)
        write_results(os.path.dirname(results_directory), 'results.txt', f'{epanet_model} | two_phased gurobi to ipopt | {ipopt_status} | {f"{ipopt_lower_bound:.2f}" if ipopt_lower_bound is not None else "-"} | {f"{ipopt_upper_bound:.2f}" if ipopt_upper_bound is not None else "-"} | {ipopt_solution_time:.2f} | - | -')

        if len(pumps) + len(valves) == 0 or ipopt_status == 'feasible': break

        AddNoGoodCutSoc(gurobi_model, gurobi_solution)

    #if ipopt_status == 'feasible':
        #GurobiOperationalModel('two_phased ipopt check', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, ipopt_solution)

    gurobi_model.dispose()
    gurobi_model_env.dispose()
    return ipopt_status, ipopt_solution, best_lower_bound, ipopt_upper_bound, time.time() - start_time

def get_period_solution(current_solution, pipes, pumps, time_step):
    return {'flow': {link: current_solution['flow'][link][time_step] for link in pipes | pumps},
            'pump_status': {pump: current_solution['pump_status'][pump][time_step] for pump in pumps},
            'valve_status': {pipe.check_valve_label: current_solution['valve_status'][pipe.check_valve_label][time_step] for pipe in pipes.values() if pipe.check_valve}}

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

        if binary_assignment in callback_data['cut_assignments']:
            return

        if len(model._set_of_links) > 0:
            flow_values = model.cbGetSolution(model._flow)
            current_solution['flow'] = {link: np.array([flow_values[link, t] for t in model._set_of_time_periods]) for link in model._set_of_links}

        up_to = callback_data['time_steps']
        ipopt_solution = {}

        ipopt_start_time = time.time()
        for time_step in range(callback_data['time_steps']):
            period_junctions, period_tanks, period_pipes, period_pumps, period_valves = callback_data['period_objects'][time_step]
            period_gurobi_solution = get_period_solution(current_solution, period_pipes, period_pumps, time_step)

            if time_step > 0:
                for label, tank in period_tanks.items():
                    tank.head_lower_bound[0] = tank.head_upper_bound[0] = period_ipopt_solution['next_head'][label]

            period_ipopt_status, period_ipopt_solution = IpoptFixedOperationalModel('ipopt fixed callback', callback_data['results_directory'], period_junctions, period_tanks, period_pipes, period_pumps, callback_data['hydraulic_timestep'], callback_data['head_loss_equation'], period_gurobi_solution)

            if period_ipopt_status == 'feasible':
                period_nodes = period_junctions | period_tanks
                for node, head in period_ipopt_solution['head'].items():
                    if node in period_tanks.keys() or period_nodes[node].segment == 'reservoir': continue
                    if head < period_nodes[node].head_lower_bound[0] - 1e-4 or head > period_nodes[node].head_upper_bound[0] + 1e-4:
                        period_ipopt_status = 'infeasible'
                        break
                for tank, head in period_ipopt_solution['next_head'].items():
                    if head < callback_data['tanks'][tank].head_lower_bound[time_step+1] - 1e-4 or head > callback_data['tanks'][tank].head_upper_bound[time_step+1] + 1e-4:
                        period_ipopt_status = 'infeasible'
                        break
                period_links = period_pipes | period_pumps
                for link, flow in period_ipopt_solution['flow'].items():
                    if flow < period_links[link].flow_lower_bound[0] - 1e-6 or flow > period_links[link].flow_upper_bound[0] + 1e-6:
                        period_ipopt_status = 'infeasible'
                        break

            if period_ipopt_status != 'feasible':
                up_to = time_step + 1
                break

            ipopt_solution[time_step] = period_ipopt_solution

        ipopt_solution_time = time.time() - ipopt_start_time
        if period_ipopt_status != 'feasible':
            callback_data['cut_assignments'].add(binary_assignment)
            AddNoGoodCutSoc(model, current_solution, lazy_flag=True, up_to=up_to)
        else:
            upper_bound = sum(callback_data['pumps'][pump].fixed_cost[t] * current_solution['pump_status'][pump][t] + callback_data['pumps'][pump].operational_cost[t] * ipopt_solution[t]['flow'].get(pump, 0.0) for t in range(callback_data['time_steps']) for pump in callback_data['pumps'])
            if callback_data['best_upper_bound'] is None or upper_bound < callback_data['best_upper_bound']:
                callback_data['best_upper_bound'], callback_data['best_ub_solution'], callback_data['status'] = upper_bound, ipopt_solution, 'feasible'

        callback_data['log_rows'].append(f'{callback_data["epanet_model"]} | ipopt fixed callback | {period_ipopt_status} | - | {f"{upper_bound:.2f}" if period_ipopt_status == "feasible" else "-"} | {ipopt_solution_time:.2f} | - | -')

def operational_problem_soc_callback(epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    period_objects = {}
    for time_step in range(time_steps):
        period_junctions, period_tanks, period_pipes, period_pumps, period_valves = deepcopy((junctions, tanks, pipes, pumps, valves))

        for valve_name, valve in list(period_valves.items()):
            if valve.valve_type != 'check valve': continue
            valve.corresponding_pipe.end_node = valve.end_node
            valve.corresponding_pipe.check_valve_label = valve_name
            valve.end_node.incoming_edges[valve.end_node.incoming_edges.index(valve)] = valve.corresponding_pipe
            del period_junctions[valve.start_node.label], period_valves[valve_name], valve.corresponding_pipe.corresponding_check_valve

        period_objects[time_step] = set_timesteps(period_junctions, period_tanks, period_pipes, period_pumps, period_valves, [time_step])

    hull_facet_start = time.time()
    find_pipe_facets(pipes, time_steps, head_loss_equation)
    find_pump_facets(pumps, time_steps)
    hull_facet_time = time.time() - hull_facet_start

    pipe_soc_outer_app(pipes, time_steps, head_loss_equation)
    pump_soc_outer_app(pumps, time_steps)

    gurobi_model, gurobi_model_env = GurobiOperationalModelSoc('callback gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation)
    gurobi_model.setParam('TimeLimit', 3600)
    gurobi_model.setParam('LazyConstraints', 1)
    #gurobi_model.setParam('PreCrush', 1)
    #gurobi_model.setParam('MIPGap', 0.0)

    gurobi_model._callback_data = {'results_directory': results_directory, 'epanet_model': epanet_model, 'period_objects': period_objects, 'time_steps': time_steps, 
                                   'tanks': tanks, 'pumps': pumps, 'hydraulic_timestep': hydraulic_timestep, 'head_loss_equation': head_loss_equation,
                                   'status': 'nosolution', 'best_upper_bound': None, 'best_ub_solution': None, 'cut_assignments': set(), 'log_rows': []}

    start_time = time.time()
    gurobi_model.optimize(gurobi_callback)
    callback_data = gurobi_model._callback_data
    final_lower_bound = gurobi_model.ObjBound if gurobi_model.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and abs(gurobi_model.ObjBound) < GRB.INFINITY else None
    for log_row in callback_data['log_rows']:
        write_results(os.path.dirname(results_directory), 'results.txt', log_row)
    gurobi_model.dispose()
    gurobi_model_env.dispose()
    return callback_data['status'], callback_data['best_ub_solution'], final_lower_bound, callback_data['best_upper_bound'], time.time() - start_time