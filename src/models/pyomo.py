import os
import time
import tempfile
import numpy as np
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import TerminationCondition, SolutionStatus

def _headbounds(model, i, t):
    return model._nodes[i].head_lower_bound[t-1], model._nodes[i].head_upper_bound[t-1]

def _flowbounds(model, i, t):
    return model._links[i].flow_lower_bound[t-1] if i not in model._set_of_pipe_families else min((pipe.flow_lower_bound[t-1] for pipe in model._pipe_families[i].options if pipe.selection_decision != 0), default=0.0), model._links[i].flow_upper_bound[t-1] if i not in model._set_of_pipe_families else max((pipe.flow_upper_bound[t-1] for pipe in model._pipe_families[i].options if pipe.selection_decision != 0), default=0.0)

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
    model._set_of_pipe_families = pyo.Set(initialize=list(model._pipe_families.keys()))
    model._incoming_edges = pyo.Set(model._set_of_nodes, initialize=lambda model, node: [])
    model._outgoing_edges = pyo.Set(model._set_of_nodes, initialize=lambda model, node: [])

    for label, link in model._links.items():
        model._incoming_edges[link.end_node.label].add(label)
        model._outgoing_edges[link.start_node.label].add(label)

def DefineDecisionVariables(model):
    model._head_indices = ([(node,t) for node in model._set_of_nodes for t in model._set_of_time_periods] + [(tank, model._time_steps + 1) for tank in model._set_of_tanks])
    model._head = pyo.Var(model._head_indices, bounds=_headbounds)
    model._flow = pyo.Var(model._set_of_links, model._set_of_time_periods, bounds=_flowbounds)
    model._pump_status = pyo.Var(model._set_of_pumps, model._set_of_time_periods, domain=pyo.Binary)
    model._valve_status = pyo.Var(model._set_of_valves, model._set_of_time_periods, domain=pyo.Binary)
    model._pipe_status = pyo.Var(model._set_of_pipes, domain=pyo.Binary) if model._problem != 'operational' else {pipe: 1.0 for pipe in model._set_of_pipes}

    for variables in (model._head, model._flow):
        for var in variables.values():
            if var.lb == var.ub:
                var.fix(var.lb)

    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].status_decisions[t-1] != -1:
                model._pump_status[pump,t].fix(model._pumps[pump].status_decisions[t-1])

    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].status_decisions[t-1] != -1:
                model._valve_status[valve,t].fix(model._valves[valve].status_decisions[t-1])

    if model._problem != 'operational':
        for pipe in model._set_of_pipes:
            if model._pipes[pipe].selection_decision != -1:
                model._pipe_status[pipe].fix(model._pipes[pipe].selection_decision)

def _objective_function(model):
    return pyo.quicksum(model._pumps[pump].fixed_cost[t-1] * model._pump_status[pump,t] + model._pumps[pump].operational_cost[t-1] * model._flow[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods) + pyo.quicksum(model._pipes[pipe].installation_cost * model._pipe_status[pipe] for pipe in model._set_of_pipes if model._pipes[pipe].installation_cost is not None)

def DefineObjective(model):
    model.OBJ = pyo.Objective(rule=_objective_function, sense=pyo.minimize)

def pipe_head_loss_rule(model, pipe_family, t):
    if all(pipe.fixed_flow[t-1] for pipe in model._pipe_families[pipe_family].options):
        return sum(pipe.resistance_coefficient * (-1 if pipe.flow_lower_bound[t-1] < 0.0 else 1) * (abs(pipe.flow_lower_bound[t-1]) ** pipe.exponent) for pipe in model._pipe_families[pipe_family].options)
    elif model._flow[pipe_family,t].lb >= 0.0:
        return pyo.quicksum(pipe.resistance_coefficient * model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) * (model._flow[pipe_family,t] ** model._pipe_families[pipe_family].exponent)
    elif model._flow[pipe_family,t].ub <= 0.0:
        return -pyo.quicksum(pipe.resistance_coefficient * model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) * ((-model._flow[pipe_family,t]) ** model._pipe_families[pipe_family].exponent)
    else:
        return pyo.quicksum(pipe.resistance_coefficient * model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) * model._flow[pipe_family,t] * ((model._flow[pipe_family,t] ** 2) ** ((model._pipe_families[pipe_family].exponent - 1) / 2))

def AddHeadLossConstraints(model):
    model._pipe_head_loss = pyo.Expression(model._set_of_pipe_families, model._set_of_time_periods, rule=pipe_head_loss_rule)
    model._cons_head_loss_l = pyo.Constraint(model._set_of_pipe_families, model._set_of_time_periods, rule=lambda model, pipe_family, t: model._head[model._pipe_families[pipe_family].start_node.label,t] - model._head[model._pipe_families[pipe_family].end_node.label,t] - model._pipe_head_loss[pipe_family,t] >= (model._pipe_families[pipe_family].start_node.head_lower_bound[t-1] - model._pipe_families[pipe_family].end_node.head_upper_bound[t-1]) * (1 - pyo.quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options)) if model._pipe_families[pipe_family].is_optional and not any(pipe.selection_decision == 1 for pipe in model._pipe_families[pipe_family].options) else model._head[model._pipe_families[pipe_family].start_node.label,t] - model._head[model._pipe_families[pipe_family].end_node.label,t] == model._pipe_head_loss[pipe_family,t])
    model._cons_head_loss_u = pyo.Constraint(model._set_of_pipe_families, model._set_of_time_periods, rule=lambda model, pipe_family, t: model._head[model._pipe_families[pipe_family].start_node.label,t] - model._head[model._pipe_families[pipe_family].end_node.label,t] - model._pipe_head_loss[pipe_family,t] <= (model._pipe_families[pipe_family].start_node.head_upper_bound[t-1] - model._pipe_families[pipe_family].end_node.head_lower_bound[t-1]) * (1 - pyo.quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options)) if model._pipe_families[pipe_family].is_optional and not any(pipe.selection_decision == 1 for pipe in model._pipe_families[pipe_family].options) else pyo.Constraint.Skip)

def AddPipeFlowConstraints(model):
    if model._problem == 'operational': return
    model._cons_pipe_family_binary = pyo.Constraint(model._set_of_pipe_families, rule=lambda model, pipe_family: sum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) <= 1 if model._pipe_families[pipe_family].is_optional else sum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) == 1)
    model._cons_pipe_family_flow_l = pyo.Constraint(model._set_of_pipe_families, model._set_of_time_periods, rule=lambda model, pipe_family, t: model._flow[pipe_family,t] >= sum(model._pipe_status[pipe.label] * pipe.flow_lower_bound[t-1] for pipe in model._pipe_families[pipe_family].options))
    model._cons_pipe_family_flow_u = pyo.Constraint(model._set_of_pipe_families, model._set_of_time_periods, rule=lambda model, pipe_family, t: sum(model._pipe_status[pipe.label] * pipe.flow_upper_bound[t-1] for pipe in model._pipe_families[pipe_family].options) >= model._flow[pipe_family,t])

def AddPumpFlowConstraints(model):
    model._cons_pump_flow_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._flow[pump,t] >= model._pumps[pump].positive_flow_lower_bound[t-1] * model._pump_status[pump,t] if model._pumps[pump].status_decisions[t-1] == -1 else pyo.Constraint.Skip)
    model._cons_pump_flow_u = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._pumps[pump].flow_upper_bound[t-1] * model._pump_status[pump,t] >= model._flow[pump,t] if model._pumps[pump].status_decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddValveFlowConstraints(model):
    model._cons_valve_flow_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._flow[valve,t] >= model._valves[valve].positive_flow_lower_bound[t-1] * model._valve_status[valve,t] if model._valves[valve].status_decisions[t-1] == -1 else pyo.Constraint.Skip)
    model._cons_valve_flow_u = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1] >= model._flow[valve,t] if model._valves[valve].status_decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddPumpHeadConstraints(model):
    model._pump_head_gain = pyo.Expression(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c))
    model._cons_pump_head_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + model._pump_head_gain[pump,t] >= (model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]) if model._pumps[pump].status_decisions[t-1] == -1 else model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + model._pump_head_gain[pump,t] == 0 if model._pumps[pump].status_decisions[t-1] == 1 else pyo.Constraint.Skip)
    model._cons_pump_head_u = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + model._pump_head_gain[pump,t] <= (model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]) if model._pumps[pump].status_decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddValveHeadConstraints(model):
    model._valve_head_loss = pyo.Expression(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._valves[valve].resistance_coefficient * (model._flow[valve,t] ** model._valves[valve].exponent))
    model._cons_valve_head_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - model._valve_head_loss[valve,t] >= (model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve,t]) if model._valves[valve].status_decisions[t-1] == -1 else model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] == model._valve_head_loss[valve,t] if model._valves[valve].status_decisions[t-1] == 1 else pyo.Constraint.Skip)
    model._cons_valve_head_u = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - model._valve_head_loss[valve,t] <= (model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve,t]) if model._valves[valve].status_decisions[t-1] == -1 else pyo.Constraint.Skip)

def AddTankBalanceConstraints(model):
    model._cons_tank_balance = pyo.Constraint(model._set_of_tanks, model._set_of_time_periods, rule=lambda model, tank, t: model._tanks[tank].tank_coef * (pyo.quicksum(model._flow[link,t] for link in model._incoming_edges[tank]) - pyo.quicksum(model._flow[link,t] for link in model._outgoing_edges[tank])) == model._head[tank,t+1] - model._head[tank,t])

def AddJunctionBalanceConstraints(model):
    model._cons_junction_balance = pyo.Constraint(model._set_of_non_reservoir_junctions, model._set_of_time_periods, rule=lambda model, junction, t: pyo.quicksum(model._flow[link,t] for link in model._incoming_edges[junction]) - pyo.quicksum(model._flow[link,t] for link in model._outgoing_edges[junction]) == model._junctions[junction].net_demand[t-1])

def PyomoModel(method, network):
    model = pyo.ConcreteModel()
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps, model._problem = network.junctions, network.tanks, network.pipes, network.pumps, network.valves, network.time_steps, network.problem
    model._nodes, model._links, model._pipe_families = network.junctions | network.tanks, network.pipe_families | network.pumps | network.valves, network.pipe_families

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddPipeFlowConstraints(model)
    AddHeadLossConstraints(model)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
    AddPumpHeadConstraints(model)
    AddValveHeadConstraints(model)
    AddTankBalanceConstraints(model)
    AddJunctionBalanceConstraints(model)

    #with open(os.path.join(network.results_directory, 'pyomo_model.txt'), 'w') as f:
    #    model.pprint(ostream=f)

    solver = pyo.SolverFactory('baron', executable='/Users/emirsener/baron-osxarm64/baron')
    solver.options['CplexLibName'] = '/Users/emirsener/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/libcplex2212.dylib'
    solver.options['MaxTime'] = network.time_limit
    solver.options['EpsR'] = network.optimality_gap
    solver.options['LPSol'] = 3

    log_path = os.path.join(network.results_directory, f'{method}_log.txt')
    start_time = time.time()
    with tempfile.TemporaryDirectory(prefix=f'{method}_', dir=network.results_directory) as scratch:
        TempfileManager.tempdir = scratch
        results = solver.solve(model, tee=True, logfile=log_path, load_solutions=False)
    TempfileManager.tempdir = None
    solution_time = time.time() - start_time

    acceptable = len(results.solution) > 0 and results.solution(0).status in [SolutionStatus.optimal, SolutionStatus.feasible]
    problem_status = 'infeasible' if results.solver.termination_condition == TerminationCondition.infeasible else 'feasible' if acceptable else 'nosolution'

    if problem_status == 'feasible':
        model.solutions.load_from(results)
        upper_bound = pyo.value(model.OBJ)
    else:
        upper_bound = None

    lower_bound = float(pyo.value(results.problem.lower_bound, exception=False)) if (problem_status != 'infeasible' and pyo.value(results.problem.lower_bound, exception=False) is not None and np.isfinite(pyo.value(results.problem.lower_bound, exception=False))) else None

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            solution['head'] = {node: np.array([pyo.value(model._head[node,t]) for t in (model._set_of_time_periods if node in model._set_of_junctions else model._set_of_time_periods_with_terminal)]) for node in model._set_of_nodes}

        if len(model._set_of_links) > 0:
            solution['flow'] = {link: np.array([pyo.value(model._flow[link,t]) for t in model._set_of_time_periods]) for link in model._set_of_links}

        if len(model._set_of_pumps) > 0:
            solution['pump_status'] = {pump: np.rint([pyo.value(model._pump_status[pump,t]) for t in model._set_of_time_periods]) for pump in model._set_of_pumps}

        if len(model._set_of_valves) > 0:
            solution['valve_status'] = {valve: np.rint([pyo.value(model._valve_status[valve,t]) for t in model._set_of_time_periods]) for valve in model._set_of_valves}

        if len(model._set_of_pipes) > 0:
            solution['pipe_status'] = {pipe: int(round(pyo.value(model._pipe_status[pipe]))) if model._problem != 'operational' else 1 for pipe in model._set_of_pipes}

    #with open(os.path.join(results_directory, f'{method}_solution.txt'), 'w') as f:
    #    model.display(ostream=f)

    return problem_status, solution, lower_bound, upper_bound, solution_time