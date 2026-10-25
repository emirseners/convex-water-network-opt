import time
import numpy as np
from gurobipy import GRB
from .hydraulics import todini_pilati
from .models.mosek import MosekModel, SolveMosekModel, AddNoGoodCutMosek
from .models.gurobi_soc import GurobiModelSoc, SolveGurobiModelSoc, AddNoGoodCutGurobi

def two_phased(network, binary_flow_direction, binary_pump_valve, binary_pipe_selection):
    mosek_model = MosekModel('two_phased mosek', network, binary_flow_direction, binary_pump_valve, binary_pipe_selection)
    mosek_model.setSolverParam("mioMaxNumSolutions", 1)

    lower_bound, upper_bound, incumbent, status = None, None, None, 'nosolution'
    incumbent_count, hydraulic_feasible_count = 0, 0
    start_time = time.time()

    while True:
        remaining_time = network.time_limit - (time.time() - start_time)
        if remaining_time <= 0: break
        mosek_model.setSolverParam('mioMaxTime', remaining_time)

        mosek_status, mosek_solution, mosek_lower_bound = SolveMosekModel(mosek_model, 'two_phased mosek', network.results_directory)
        if mosek_lower_bound is not None and (lower_bound is None or mosek_lower_bound > lower_bound): lower_bound = mosek_lower_bound

        if mosek_status != 'feasible' or mosek_solution is None:
            if mosek_status == 'infeasible' and upper_bound is not None:
                lower_bound = upper_bound
            break

        incumbent_count += 1
        hydraulic_solution, hydraulic_status, up_to = todini_pilati(network, mosek_solution)
        AddNoGoodCutMosek(mosek_model, mosek_solution, up_to=up_to)

        if hydraulic_status == 'feasible':
            hydraulic_feasible_count += 1
            candidate_upper_bound = sum(network.pumps[pump].fixed_cost[t] * hydraulic_solution['pump_status'][pump][t] + network.pumps[pump].operational_cost[t] * hydraulic_solution['flow'][pump][t] for t in range(network.time_steps) for pump in network.pumps)
            if network.problem != 'operational':
                candidate_upper_bound += sum(pipe.installation_cost * hydraulic_solution['pipe_status'][label] for label, pipe in network.pipes.items() if pipe.installation_cost is not None)
            if upper_bound is None or candidate_upper_bound < upper_bound:
                upper_bound, incumbent, status = candidate_upper_bound, hydraulic_solution, 'feasible'
                mosek_model.setSolverParam("upperObjCut", candidate_upper_bound)

        if lower_bound is not None and upper_bound is not None:
            if (upper_bound - lower_bound) <= network.optimality_gap * upper_bound: break

    if lower_bound is not None and upper_bound is not None:
        lower_bound = min(lower_bound, upper_bound)

    mosek_model.dispose()
    return status, incumbent, lower_bound, upper_bound, time.time() - start_time, incumbent_count, hydraulic_feasible_count

def two_phased_soc(network, binary_flow_direction, binary_pump_valve, binary_pipe_selection):
    gurobi_model, gurobi_model_env = GurobiModelSoc('two_phased gurobi', network, binary_flow_direction, binary_pump_valve, binary_pipe_selection)
    gurobi_model.setParam('SolutionLimit', 1)

    lower_bound, upper_bound, incumbent, status = None, None, None, 'nosolution'
    incumbent_count, hydraulic_feasible_count = 0, 0
    start_time = time.time()

    while True:
        remaining_time = network.time_limit - (time.time() - start_time)
        if remaining_time <= 0: break
        gurobi_model.setParam('TimeLimit', remaining_time)

        gurobi_status, gurobi_solution, gurobi_lower_bound = SolveGurobiModelSoc(gurobi_model, 'two_phased gurobi', network.results_directory)
        if gurobi_lower_bound is not None and (lower_bound is None or gurobi_lower_bound > lower_bound): lower_bound = gurobi_lower_bound

        if gurobi_status != 'feasible' or gurobi_solution is None:
            if gurobi_model.Status == GRB.INFEASIBLE and upper_bound is not None:
                lower_bound = upper_bound
            break

        incumbent_count += 1
        hydraulic_solution, hydraulic_status, up_to = todini_pilati(network, gurobi_solution)
        AddNoGoodCutGurobi(gurobi_model, gurobi_solution, up_to=up_to)

        if hydraulic_status == 'feasible':
            hydraulic_feasible_count += 1
            candidate_upper_bound = sum(network.pumps[pump].fixed_cost[t] * hydraulic_solution['pump_status'][pump][t] + network.pumps[pump].operational_cost[t] * hydraulic_solution['flow'][pump][t] for t in range(network.time_steps) for pump in network.pumps)
            if network.problem != 'operational':
                candidate_upper_bound += sum(pipe.installation_cost * hydraulic_solution['pipe_status'][label] for label, pipe in network.pipes.items() if pipe.installation_cost is not None)
            if upper_bound is None or candidate_upper_bound < upper_bound:
                upper_bound, incumbent, status = candidate_upper_bound, hydraulic_solution, 'feasible'
                gurobi_model.setParam('Cutoff', candidate_upper_bound)

        if lower_bound is not None and upper_bound is not None:
            if (upper_bound - lower_bound) <= network.optimality_gap * upper_bound: break

    if lower_bound is not None and upper_bound is not None:
        lower_bound = min(lower_bound, upper_bound)

    gurobi_model.dispose()
    gurobi_model_env.dispose()
    return status, incumbent, lower_bound, upper_bound, time.time() - start_time, incumbent_count, hydraulic_feasible_count

def gurobi_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        callback_data = model._callback_data
        network = callback_data['network']
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

        if model._problem != 'operational' and len(model._set_of_pipes) > 0:
            pipe_values = model.cbGetSolution(model._pipe_status)
            current_solution['pipe_status'] = {pipe: int(round(pipe_values[pipe])) for pipe in model._set_of_pipes}
            binary_assignment = binary_assignment + tuple(current_solution['pipe_status'][pipe] for pipe in model._set_of_pipes)
        else:
            current_solution['pipe_status'] = {pipe: 1 for pipe in model._set_of_pipes}

        if binary_assignment in callback_data['cut_assignments'].keys():
            AddNoGoodCutGurobi(model, current_solution, lazy_flag=True, up_to=callback_data['cut_assignments'][binary_assignment])
            return

        if len(model._set_of_links) > 0:
            flow_values = model.cbGetSolution(model._flow)
            current_solution['flow'] = {link: np.array([flow_values[link, t] for t in model._set_of_time_periods]) for link in model._set_of_links}

        if len(model._set_of_nodes) > 0:
            head_values = model.cbGetSolution(model._head)
            current_solution['head'] = {node: np.array([head_values[node, t] for t in model._set_of_time_periods]) for node in model._set_of_nodes}

        callback_data['incumbent_count'] += 1
        hydraulic_solution, hydraulic_status, up_to = todini_pilati(network, current_solution)
        callback_data['cut_assignments'][binary_assignment] = up_to
        AddNoGoodCutGurobi(model, current_solution, lazy_flag=True, up_to=up_to)

        if hydraulic_status == 'feasible':
            callback_data['hydraulic_feasible_count'] += 1
            candidate_upper_bound = sum(network.pumps[pump].fixed_cost[t] * hydraulic_solution['pump_status'][pump][t] + network.pumps[pump].operational_cost[t] * hydraulic_solution['flow'][pump][t] for t in range(network.time_steps) for pump in network.pumps)
            if network.problem != 'operational':
                candidate_upper_bound += sum(pipe.installation_cost * hydraulic_solution['pipe_status'][label] for label, pipe in network.pipes.items() if pipe.installation_cost is not None)
            if callback_data['upper_bound'] is None or candidate_upper_bound < callback_data['upper_bound']:
                callback_data['upper_bound'], callback_data['incumbent'], callback_data['status'] = candidate_upper_bound, hydraulic_solution, 'feasible'

        lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        if callback_data['upper_bound'] is not None and lower_bound < GRB.INFINITY:
            if callback_data['upper_bound'] - lower_bound <= network.optimality_gap * callback_data['upper_bound']: model.terminate()

def callback_soc(network, binary_flow_direction, binary_pump_valve, binary_pipe_selection):
    gurobi_model, gurobi_model_env = GurobiModelSoc('callback gurobi', network, binary_flow_direction, binary_pump_valve, binary_pipe_selection)
    gurobi_model.setParam('TimeLimit', network.time_limit)
    gurobi_model.setParam('LazyConstraints', 1)
    #gurobi_model.setParam('MIPFocus', 1)
    #gurobi_model.setParam('PreCrush', 1)
    #gurobi_model.setParam('MIPGap', 0.0)

    gurobi_model._callback_data = {'network': network, 'status': 'nosolution', 'upper_bound': None, 'incumbent': None, 'cut_assignments': {}, 'incumbent_count': 0, 'hydraulic_feasible_count': 0}

    start_time = time.time()
    gurobi_model.optimize(gurobi_callback)
    callback_data = gurobi_model._callback_data
    lower_bound = gurobi_model.ObjBound if gurobi_model.Status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) and abs(gurobi_model.ObjBound) < GRB.INFINITY else None
    if (gurobi_model.Status == GRB.INFEASIBLE and callback_data['upper_bound'] is not None):
        lower_bound = callback_data['upper_bound']
    if lower_bound is not None and callback_data['upper_bound'] is not None:
        lower_bound = min(lower_bound, callback_data['upper_bound'])

    gurobi_model.dispose()
    gurobi_model_env.dispose()
    return callback_data['status'], callback_data['incumbent'], lower_bound, callback_data['upper_bound'], time.time() - start_time, callback_data['incumbent_count'], callback_data['hydraulic_feasible_count']
