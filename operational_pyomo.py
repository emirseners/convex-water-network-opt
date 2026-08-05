import os
import time
import tempfile
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.common.tempfiles import TempfileManager

def _headbounds(model, i, t):
    return model._nodes[i].head_lower_bound[t-1], model._nodes[i].head_upper_bound[t-1]

def _flowbounds(model, i, t):
    return model._links[i].flow_lower_bound[t-1], model._links[i].flow_upper_bound[t-1]

def _storage_bounds(model, i, t):
    return model._tanks[i].storage_lower_bound[t-1], model._tanks[i].storage_upper_bound[t-1]

def _pump_flow_head_loss_bounds(model, i, t):
    return -model._pumps[i].curve_b * (model._pumps[i].flow_lower_bound[t-1] ** model._pumps[i].curve_c), -model._pumps[i].curve_b * (model._pumps[i].flow_upper_bound[t-1] ** model._pumps[i].curve_c)

def _objective_function(model):
    return pyo.quicksum((model._pumps[i].fixed_cost[t-1] * model._pump_status[i,t]) + (model._pumps[i].operational_cost[t-1] * model._flow[i,t]) for i in model._set_of_pumps for t in model._set_of_time_periods)

def DefineSets(model):
    model._set_of_junctions = pyo.Set(initialize=list(model._junctions.keys()))
    model._set_of_non_reservoir_junctions = pyo.Set(initialize=[label for label, junction in model._junctions.items() if junction.segment != 'reservoir'])
    model._set_of_tanks = pyo.Set(initialize=list(model._tanks.keys()))
    model._set_of_pipes = pyo.Set(initialize=list(model._pipes.keys()))
    model._set_of_pumps = pyo.Set(initialize=list(model._pumps.keys()))
    model._set_of_valves = pyo.Set(initialize=list(model._valves.keys()))
    model._set_of_time_periods = pyo.RangeSet(1, model._time_steps)
    model._set_of_time_periods_with_terminal = pyo.RangeSet(1, model._time_steps + 1)
    model._set_of_nodes = pyo.Set(initialize=list(model._nodes.keys()))
    model._set_of_links = pyo.Set(initialize=list(model._links.keys()))

def DefineDecisionVariables(model, method, flow_direction, epsilon, initial_solution=None):
    model._head = pyo.Var(model._set_of_nodes, model._set_of_time_periods, bounds=_headbounds)
    model._flow = pyo.Var(model._set_of_links, model._set_of_time_periods, bounds=_flowbounds)
    model._storage = pyo.Var(model._set_of_tanks, model._set_of_time_periods_with_terminal, bounds=_storage_bounds)
    model._pump_flow_head_loss = pyo.Var(model._set_of_pumps, model._set_of_time_periods, bounds=_pump_flow_head_loss_bounds)

    for node in model._set_of_nodes:
        for t in model._set_of_time_periods:
            lower, upper = _headbounds(model, node, t)
            if lower == upper:
                model._head[node,t].fix(lower)

    for link in model._set_of_links:
        for t in model._set_of_time_periods:
            lower, upper = _flowbounds(model, link, t)
            if lower == upper:
                model._flow[link,t].fix(lower)

    for storage in model._set_of_tanks:
        for t in model._set_of_time_periods_with_terminal:
            lower, upper = _storage_bounds(model, storage, t)
            if lower == upper:
                model._storage[storage,t].fix(lower)

    if initial_solution is not None:
        model._pump_status = pyo.Param(model._set_of_pumps, model._set_of_time_periods, initialize={(pump, t): float(initial_solution['pump_status'][pump][t-1]) for pump in model._set_of_pumps for t in model._set_of_time_periods}, within=pyo.Binary)
        model._valve_status = pyo.Param(model._set_of_valves, model._set_of_time_periods, initialize={(valve, t): float(initial_solution['valve_status'][valve][t-1]) for valve in model._set_of_valves for t in model._set_of_time_periods}, within=pyo.Binary)

        for key, solutions in initial_solution.items():
            if key == 'head':
                for node, heads in solutions.items():
                    for t in model._set_of_time_periods:
                        model._head[node,t].value = min(max(heads[t-1], model._nodes[node].head_lower_bound[t-1]), model._nodes[node].head_upper_bound[t-1])

            elif key == 'flow':
                for link, flows in solutions.items():
                    for t in model._set_of_time_periods:
                        if link in model._set_of_pumps:
                            if model._pump_status[link,t] == 1:
                                model._flow[link,t].setlb(max(model._pumps[link].flow_lower_bound[t-1], epsilon))
                                model._flow[link,t].value = min(max(flows[t-1], model._flow[link,t].lb), model._flow[link,t].ub) #min(max(epsilon, flows[t-1], model._links[link].flow_lower_bound[t-1]), model._links[link].flow_upper_bound[t-1])
                            else:
                                model._flow[link,t].fix(0.0)
                        elif link in model._set_of_valves:
                            if model._valve_status[link,t] == 0:
                                model._flow[link,t].fix(0.0)
                            else:
                                model._flow[link,t].value = min(max(flows[t-1], model._links[link].flow_lower_bound[t-1]), model._links[link].flow_upper_bound[t-1])
                        else:
                            if model._pipes[link].flow_lower_bound[t-1] != model._pipes[link].flow_upper_bound[t-1] and flow_direction:
                                if model._pipes[link].flow_lower_bound[t-1] >= 0.0 or (model._pipes[link].flow_upper_bound[t-1] > 0.0 and flows[t-1] >= 0.0):
                                    model._flow[link,t].setlb(max(model._pipes[link].flow_lower_bound[t-1], epsilon))
                                else:
                                    model._flow[link,t].setub(min(model._pipes[link].flow_upper_bound[t-1], -epsilon))
                            model._flow[link,t].value = min(max(flows[t-1], model._flow[link,t].lb), model._flow[link,t].ub) #min(max(epsilon, flows[t-1], model._links[link].flow_lower_bound[t-1]), model._links[link].flow_upper_bound[t-1])

            elif key == 'storage':
                for tank, storages in solutions.items():
                    for t in model._set_of_time_periods_with_terminal:
                        lower, upper = _storage_bounds(model, tank, t)
                        model._storage[tank,t].value = min(max(storages[t-1], lower), upper)

    else:
        model._pump_status = pyo.Var(model._set_of_pumps, model._set_of_time_periods, domain=pyo.UnitInterval if method in ['two_phased', 'ipopt'] else pyo.Binary)
        model._valve_status = pyo.Var(model._set_of_valves, model._set_of_time_periods, domain=pyo.UnitInterval if method in ['two_phased', 'ipopt'] else pyo.Binary)
        for pump in model._set_of_pumps:
            for t in model._set_of_time_periods:
                if model._pumps[pump].decisions[t-1] == 1:
                    model._pump_status[pump,t].fix(1)
                    model._flow[pump,t].setlb(max(model._pumps[pump].flow_lower_bound[t-1], epsilon))
                elif model._pumps[pump].decisions[t-1] == 0:
                    model._pump_status[pump,t].fix(0)
                    model._flow[pump,t].fix(0.0)

        for valve in model._set_of_valves:
            for t in model._set_of_time_periods:
                if model._valves[valve].decisions[t-1] == 1:
                    model._valve_status[valve,t].fix(1)
                elif model._valves[valve].decisions[t-1] == 0:
                    model._valve_status[valve,t].fix(0)
                    model._flow[valve,t].fix(0.0)

def DefineObjective(model):
    model.OBJ = pyo.Objective(rule=_objective_function, sense=pyo.minimize)

def headloss_rule(model, pipe, t, epsilon, exponent):
    if model._pipes[pipe].flow_lower_bound[t-1] == model._pipes[pipe].flow_upper_bound[t-1]:
        return model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == model._pipes[pipe].loss_coefficient * (-1 if model._pipes[pipe].flow_lower_bound[t-1] < 0.0 else 1) * (abs(model._pipes[pipe].flow_lower_bound[t-1]) ** exponent)
    elif model._flow[pipe,t].lb > 0.0:
        return model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == model._pipes[pipe].loss_coefficient * (model._flow[pipe,t] ** exponent)
    elif model._flow[pipe,t].ub < 0.0:
        return model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == -model._pipes[pipe].loss_coefficient * ((-model._flow[pipe,t]) ** exponent)
    else:
        return model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == model._pipes[pipe].loss_coefficient * model._flow[pipe,t] * (((model._flow[pipe,t] ** 2) + epsilon) ** ((exponent - 1) / 2))

def AddHeadLossConstraints(model, head_loss_equation, epsilon):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    model._cons_head_loss = pyo.Constraint(model._set_of_pipes, model._set_of_time_periods, rule=lambda model, pipe, t: headloss_rule(model, pipe, t, epsilon, exponent))

def AddPumpFlowConstraints(model):
    model._cons_pump_flow_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._flow[pump,t] >= model._pumps[pump].flow_lower_bound[t-1] * model._pump_status[pump,t] if model._pumps[pump].decisions[t-1] == -1 else pyo.Constraint.Skip)
    model._cons_pump_flow_u = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._pumps[pump].flow_upper_bound[t-1] * model._pump_status[pump,t] >= model._flow[pump,t] if model._pumps[pump].decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddValveFlowConstraints(model):
    model._cons_valve_flow_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._flow[valve,t] >= model._valves[valve].flow_lower_bound[t-1] * model._valve_status[valve,t] if model._valves[valve].decisions[t-1] == -1 else pyo.Constraint.Skip)
    model._cons_valve_flow_u = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1] >= model._flow[valve,t] if model._valves[valve].decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddPumpHeadConstraints(model, initial_solution, epsilon):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].decisions[t-1] == 0 or (initial_solution is not None and initial_solution['pump_status'][pump][t-1] == 0):
                model._pump_flow_head_loss[pump,t].fix(0.0)
            elif model._pumps[pump].flow_lower_bound[t-1] == model._pumps[pump].flow_upper_bound[t-1]:
                model._pump_flow_head_loss[pump,t].fix(-model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c))
            elif initial_solution is not None:
                model._pump_flow_head_loss[pump,t].value = -model._pumps[pump].curve_b * (model._flow[pump,t].value ** model._pumps[pump].curve_c)
            elif model._pumps[pump].decisions[t-1] != 1 and model._pumps[pump].curve_c != 1:
                model._pump_flow_head_loss[pump,t].setlb(-model._pumps[pump].curve_b * (((model._pumps[pump].flow_lower_bound[t-1] + epsilon) ** model._pumps[pump].curve_c) - (epsilon ** model._pumps[pump].curve_c)))
                model._pump_flow_head_loss[pump,t].setub(-model._pumps[pump].curve_b * (((model._pumps[pump].flow_upper_bound[t-1] + epsilon) ** model._pumps[pump].curve_c) - (epsilon ** model._pumps[pump].curve_c)))
    if initial_solution is not None:
        model._cons_pump_flow_head_loss = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._pump_flow_head_loss[pump,t] == -model._pumps[pump].curve_b * model._flow[pump,t] if initial_solution['pump_status'][pump][t-1] == 1 and model._pumps[pump].curve_c == 1 and model._pumps[pump].flow_lower_bound[t-1] != model._pumps[pump].flow_upper_bound[t-1] else model._pump_flow_head_loss[pump,t] == -model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c) if initial_solution['pump_status'][pump][t-1] == 1 and model._pumps[pump].flow_lower_bound[t-1] != model._pumps[pump].flow_upper_bound[t-1] else pyo.Constraint.Skip)
        model._cons_pump_head_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a - model._pump_flow_head_loss[pump,t] == 0 if initial_solution['pump_status'][pump][t-1] == 1 else pyo.Constraint.Skip)
    else:
        model._cons_pump_flow_head_loss = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._pump_flow_head_loss[pump,t] == -model._pumps[pump].curve_b * model._flow[pump,t] if model._pumps[pump].decisions[t-1] != 0 and model._pumps[pump].curve_c == 1 and model._pumps[pump].flow_lower_bound[t-1] != model._pumps[pump].flow_upper_bound[t-1] else model._pump_flow_head_loss[pump,t] == -model._pumps[pump].curve_b * (((model._flow[pump,t] + epsilon) ** model._pumps[pump].curve_c) - (epsilon ** model._pumps[pump].curve_c)) if model._pumps[pump].decisions[t-1] == -1 and model._pumps[pump].flow_lower_bound[t-1] != model._pumps[pump].flow_upper_bound[t-1] else model._pump_flow_head_loss[pump,t] == -model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c) if model._pumps[pump].decisions[t-1] == 1 and model._pumps[pump].flow_lower_bound[t-1] != model._pumps[pump].flow_upper_bound[t-1] else pyo.Constraint.Skip)
        model._cons_pump_head_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) - model._pump_flow_head_loss[pump,t] >= (model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]) if model._pumps[pump].decisions[t-1] == -1 else model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a - model._pump_flow_head_loss[pump,t] == 0 if model._pumps[pump].decisions[t-1] == 1 else pyo.Constraint.Skip)
        model._cons_pump_head_u = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) - model._pump_flow_head_loss[pump,t] <= (model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]) if model._pumps[pump].decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddValveHeadConstraints(model, initial_solution):
    if initial_solution is not None:
        model._cons_valve_head_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._head[model._valves[valve].start_node.label,t] == model._head[model._valves[valve].end_node.label,t] if initial_solution['valve_status'][valve][t-1] == 1 else pyo.Constraint.Skip)
    else:
        model._cons_valve_head_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] >= (model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve,t]) if model._valves[valve].decisions[t-1] == -1 else model._head[model._valves[valve].start_node.label,t] == model._head[model._valves[valve].end_node.label,t] if model._valves[valve].decisions[t-1] == 1 else pyo.Constraint.Skip)
        model._cons_valve_head_u = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] <= (model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve,t]) if model._valves[valve].decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddTankBalanceConstraints(model, hydraulic_timestep): #m^3 / s == m^3
    model._cons_tank_balance = pyo.Constraint(model._set_of_tanks, model._set_of_time_periods, rule=lambda model, tank, t: hydraulic_timestep * sum(model._flow[edge.label,t] for edge in model._tanks[tank].incoming_edges) - hydraulic_timestep * sum(model._flow[edge.label,t] for edge in model._tanks[tank].outgoing_edges) == model._storage[tank,t+1] - model._storage[tank,t])

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    model._cons_junction_balance = pyo.Constraint(model._set_of_non_reservoir_junctions, model._set_of_time_periods, rule=lambda model, junction, t: sum(model._flow[edge.label,t] for edge in model._junctions[junction].incoming_edges) - sum(model._flow[edge.label,t] for edge in model._junctions[junction].outgoing_edges) == model._junctions[junction].net_demand[t-1])

def AddTankHeadConstraints(model):
    model._cons_tank_head = pyo.Constraint(model._set_of_tanks, model._set_of_time_periods, rule=lambda model, tank, t: model._head[tank,t] == model._tanks[tank].elevation + ((1 / model._tanks[tank].area) * model._storage[tank,t]) if t > 1 else pyo.Constraint.Skip)

def PyomoOperationalModel(method, epanet_model, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation, initial_solution = None, flow_direction = False, epsilon = 1e-12):
    model = pyo.ConcreteModel()
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model, method, flow_direction, epsilon, initial_solution)
    DefineObjective(model)
    AddHeadLossConstraints(model, head_loss_equation, epsilon)
    if initial_solution is None:
        AddPumpFlowConstraints(model)
        AddValveFlowConstraints(model)
    AddPumpHeadConstraints(model, initial_solution, epsilon)
    AddValveHeadConstraints(model, initial_solution)
    AddTankBalanceConstraints(model, hydraulic_timestep)
    AddJunctionBalanceConstraints(model)
    AddTankHeadConstraints(model)

    #with open(os.path.join(results_directory, 'pyomo_model.txt'), 'w') as f:
    #    model.pprint(ostream=f)

    if method in ['two_phased', 'ipopt']:
        solver = pyo.SolverFactory('ipopt', executable=os.path.expanduser('~/.idaes/bin/ipopt'))
        solver.options['max_cpu_time'] = 600
        solver.options['bound_relax_factor'] = 0
        #solver.options['bound_push'] = 1e-6
        #solver.options['bound_frac'] = 1e-6

    else:
        solver = pyo.SolverFactory('baron', executable='/Users/emirsener/baron-osxarm64/baron')
        solver.options['CplexLibName'] = '/Users/emirsener/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/libcplex2212.dylib'
        solver.options['MaxTime'] = 600
        solver.options['LPSol'] = 3

    log_path = os.path.join(results_directory, f'{epanet_model}_log.txt')
    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix=f'{epanet_model}_', dir=results_directory) as scratch:
        TempfileManager.tempdir = scratch
        results = solver.solve(model, tee=True, logfile=log_path, load_solutions=False)
    solution_time = time.time() - start_time

    acceptable = results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible]
    problem_status = 'infeasible' if results.solver.termination_condition == TerminationCondition.infeasible else 'feasible' if (acceptable and len(results.solution) > 0) else 'nosolution'

    if problem_status == 'feasible':
        model.solutions.load_from(results)
        upper_bound = pyo.value(model.OBJ)
    else:
        upper_bound = None

    lower_bound = None if method in ['two_phased', 'ipopt'] else float(pyo.value(results.problem.lower_bound, exception=False)) if (problem_status != 'infeasible' and pyo.value(results.problem.lower_bound, exception=False) is not None and np.isfinite(pyo.value(results.problem.lower_bound, exception=False))) else None

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            solution['head'] = {node: np.array([pyo.value(model._head[node, t]) for t in model._set_of_time_periods]) for node in model._set_of_nodes}

        if len(model._set_of_links) > 0:
            solution['flow'] = {link: np.array([pyo.value(model._flow[link, t]) for t in model._set_of_time_periods]) for link in model._set_of_links}

        if len(model._set_of_tanks) > 0:
            solution['storage'] = {tank: np.array([pyo.value(model._storage[tank, t]) for t in model._set_of_time_periods_with_terminal]) for tank in model._set_of_tanks}

        if len(model._set_of_pumps) > 0:
            solution['pump_status'] = {pump: np.array([pyo.value(model._pump_status[pump, t]) for t in model._set_of_time_periods]) for pump in model._set_of_pumps}

        if len(model._set_of_valves) > 0:
            solution['valve_status'] = {valve: np.array([pyo.value(model._valve_status[valve, t]) for t in model._set_of_time_periods]) for valve in model._set_of_valves}

    #with open(os.path.join(results_directory, f'{epanet_model}_solution.txt'), 'w') as f:
    #    model.display(ostream=f)

    return problem_status, solution, lower_bound, upper_bound, solution_time