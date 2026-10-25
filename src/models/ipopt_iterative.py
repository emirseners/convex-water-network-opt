'''
It is not being used for now.
'''

import os
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from pyomo.common.tempfiles import TempfileManager

def _objective_function(model):
    return (pyo.quicksum(((-model._pumps[pump].curve_b / (model._pumps[pump].curve_c + 1)) * (model._flow[pump] ** (model._pumps[pump].curve_c + 1))) - model._pumps[pump].curve_a * model._flow[pump] for pump in model._set_of_pumps)
          + pyo.quicksum((model._pipes[pipe].resistance_coefficient / (model._pipes[pipe].exponent + 1)) * abs(model._flow[pipe]) ** (model._pipes[pipe].exponent + 1) for pipe in model._set_of_pipes)
          + pyo.quicksum((model._valves[valve].resistance_coefficient / (model._valves[valve].exponent + 1)) * abs(model._flow[valve]) ** (model._valves[valve].exponent + 1) for valve in model._set_of_valves)
          + pyo.quicksum(model._junctions[reservoir].head_lower_bound[0] * (sum(model._flow[edge.label] for edge in model._junctions[reservoir].incoming_edges if edge.label in model._links) - sum(model._flow[edge.label] for edge in model._junctions[reservoir].outgoing_edges if edge.label in model._links)) for reservoir in model._set_of_reservoirs)
          + pyo.quicksum(model._tanks[tank].head_lower_bound[0] * (sum(model._flow[edge.label] for edge in model._tanks[tank].incoming_edges if edge.label in model._links) - sum(model._flow[edge.label] for edge in model._tanks[tank].outgoing_edges if edge.label in model._links)) for tank in model._set_of_tanks)
           )

def DefineSets(model):
    model._set_of_junctions = pyo.Set(initialize=list(model._junctions.keys()))
    model._set_of_non_reservoir_junctions = pyo.Set(initialize=[label for label, junction in model._junctions.items() if junction.segment != 'reservoir'])
    model._set_of_reservoirs = pyo.Set(initialize=[label for label, junction in model._junctions.items() if junction.segment == 'reservoir'])
    model._set_of_connected_junctions = pyo.Set(initialize=[label for label, junction in model._junctions.items() if junction.segment != 'reservoir' and any(edge.label in model._links for edge in junction.incoming_edges + junction.outgoing_edges)])
    model._set_of_tanks = pyo.Set(initialize=list(model._tanks.keys()))
    model._set_of_pipes = pyo.Set(initialize=list(model._pipes.keys()))
    model._set_of_valves = pyo.Set(initialize=list(model._valves.keys()))
    model._set_of_pumps = pyo.Set(initialize=list(model._pumps.keys()))
    model._set_of_nodes = pyo.Set(initialize=list(model._nodes.keys()))
    model._set_of_links = pyo.Set(initialize=list(model._links.keys()))

def DefineDecisionVariables(model, gurobi_solution):
    model._flow = pyo.Var(model._set_of_links)
    for pump in model._set_of_pumps:
        model._flow[pump].setlb(0.0)
    for link, initial_value in gurobi_solution['flow'].items():
        if link not in model._links: continue
        model._flow[link].value = initial_value

def DefineObjective(model):
    model.OBJ = pyo.Objective(rule=_objective_function, sense=pyo.minimize)

def AddJunctionBalanceConstraints(model):
    model._cons_junction_balance = pyo.Constraint(model._set_of_connected_junctions, rule=lambda model, junction: sum(model._flow[edge.label] for edge in model._junctions[junction].incoming_edges if edge.label in model._links) - sum(model._flow[edge.label] for edge in model._junctions[junction].outgoing_edges if edge.label in model._links) == model._junctions[junction].net_demand[0])

def IpoptFixedModel(model_name, results_directory, junctions, tanks, pipes, pumps, valves, gurobi_solution):
    model = pyo.ConcreteModel()
    model._junctions, model._tanks = junctions, tanks
    model._pipes = {label: pipe for label, pipe in pipes.items()}
    model._pumps = {label: pump for label, pump in pumps.items() if gurobi_solution['pump_status'][label] == 1}
    model._valves = {label: valve for label, valve in valves.items() if gurobi_solution['valve_status'][label] == 1}
    model._nodes, model._links = junctions | tanks, model._pipes | model._pumps | model._valves

    DefineSets(model)
    DefineDecisionVariables(model, gurobi_solution)
    DefineObjective(model)
    AddJunctionBalanceConstraints(model)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    solver = pyo.SolverFactory('ipopt', executable=os.path.expanduser('~/.idaes/bin/ipopt'))
    solver.options['bound_relax_factor'] = 0
    solver.options['dual_inf_tol'] = 1e-4

    log_path = os.path.join(results_directory, f'{model_name}_log.txt')
    TempfileManager.tempdir = None
    results = solver.solve(model, tee=False, logfile=log_path, load_solutions=False)

    acceptable = results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible]
    problem_status = 'infeasible' if results.solver.termination_condition == TerminationCondition.infeasible else 'feasible' if (acceptable and len(results.solution) > 0) else 'nosolution'

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        model.solutions.load_from(results)

        if len(model._set_of_nodes) > 0:
            solution['head'] = {junction: -model.dual[model._cons_junction_balance[junction]] for junction in model._set_of_connected_junctions} | {reservoir: model._junctions[reservoir].head_lower_bound[0] for reservoir in model._set_of_reservoirs} | {tank: model._tanks[tank].head_lower_bound[0] for tank in model._set_of_tanks}

        if len(model._set_of_links) > 0:
            solution['flow'] = {link: pyo.value(model._flow[link]) for link in model._set_of_links}

            if len(model._set_of_tanks) > 0:
                solution['next_head'] = {tank: solution['head'][tank] + (model._tanks[tank].tank_coef * (sum(solution['flow'][edge.label] for edge in model._tanks[tank].incoming_edges if edge.label in model._links) - sum(solution['flow'][edge.label] for edge in model._tanks[tank].outgoing_edges if edge.label in model._links))) for tank in model._set_of_tanks}

    return problem_status, solution