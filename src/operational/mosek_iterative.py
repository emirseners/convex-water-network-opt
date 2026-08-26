"""
Not complete
"""

import os
import numpy as np
from mosek.fusion import *
import mosek.fusion.pythonic

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_reservoirs = list(model._reservoirs.keys())
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_links = list(model._links.keys())
    model._node_index = {label: index for index, label in enumerate(model._set_of_nodes)}
    model._link_index = {label: index for index, label in enumerate(model._set_of_links)}

def DefineDecisionVariables(model, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    if model._set_of_links:
        model._flow = model.variable('flow', len(model._set_of_links), Domain.inRange(
            np.stack([model._links[l].flow_lower_bound[0] for l in model._set_of_links]), np.stack([model._links[l].flow_upper_bound[0] for l in model._set_of_links])))

    if model._set_of_pipes:
        model._phi_pipe = model.variable('phi_pipe', len(model._set_of_pipes), Domain.inRange(
            np.stack([0.0 if model._pipes[pipe].flow_lower_bound[0] <= 0 and model._pipes[pipe].flow_upper_bound[0] >= 0 else min(abs(model._pipes[pipe].flow_lower_bound[0]), abs(model._pipes[pipe].flow_upper_bound[0])) ** (exponent + 1) for pipe in model._set_of_pipes]), np.stack([max(abs(model._pipes[pipe].flow_lower_bound[0]), abs(model._pipes[pipe].flow_upper_bound[0])) ** (exponent + 1) for pipe in model._set_of_pipes])))

    if model._set_of_pumps:
        model._phi_pump = model.variable('phi_pump', len(model._set_of_pumps), Domain.inRange(
            np.stack([model._pumps[pump].flow_lower_bound[0] ** (model._pumps[pump].curve_c + 1) for pump in model._set_of_pumps]), np.stack([model._pumps[pump].flow_upper_bound[0] ** (model._pumps[pump].curve_c + 1) for pump in model._set_of_pumps])))

def DefineObjective(model, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2

    flow_coefficients = np.zeros(len(model._set_of_links))
    for pump in model._set_of_pumps:
        flow_coefficients[model._link_index[pump]] -= model._pumps[pump].curve_a
    for tank in model._set_of_tanks:
        flow_coefficients += model._tanks[tank].head_lower_bound[0] * model._incidence[tank]
    for reservoir in model._set_of_reservoirs:
        flow_coefficients += model._reservoirs[reservoir].head_lower_bound[0] * model._incidence[reservoir]

    objective = Expr.dot(flow_coefficients, model._flow)
    if model._set_of_pipes:
        objective = objective + Expr.dot(np.stack([model._pipes[pipe].loss_coefficient / (exponent + 1) for pipe in model._set_of_pipes]), model._phi_pipe)
    if model._set_of_pumps:
        objective = objective + Expr.dot(np.stack([-model._pumps[pump].curve_b / (model._pumps[pump].curve_c + 1) for pump in model._set_of_pumps]), model._phi_pump)

    model.objective('objective', ObjectiveSense.Minimize, objective)

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    model._balance_cons = {}
    for junction in model._junctions.values():
        if junction.segment == 'reservoir':
            continue
        cons = model.constraint(f'cons_junction_balance_{junction.label}', Expr.dot(model._incidence[junction.label], model._flow) - junction.net_demand[0] == 0)
        model._balance_cons[junction.label] = cons

def AddPowerConeConstraints(model, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for idx, pipe in enumerate(model._set_of_pipes):
        model.constraint(Expr.vstack(model._phi_pipe[idx], 1, model._flow[model._link_index[pipe]]), Domain.inPPowerCone(1/(exponent + 1)))
    for idx, pump in enumerate(model._set_of_pumps):
        model.constraint(Expr.vstack(model._phi_pump[idx], 1, model._flow[model._link_index[pump]]), Domain.inPPowerCone(1/(model._pumps[pump].curve_c + 1)))

def DefineIncidenceMatrix(model):
    model._incidence = {}
    for node in model._nodes.values():
        incidence_vector = np.zeros(len(model._set_of_links))
        for edge in node.incoming_edges:
            incidence_vector[model._link_index[edge.label]] += 1.0
        for edge in node.outgoing_edges:
            incidence_vector[model._link_index[edge.label]] -= 1.0
        model._incidence[node.label] = incidence_vector

def FixStatus(model, gurobi_solution):
    for pump, status in gurobi_solution['pump_status'].items():
        if status == 0:
            model.constraint(f'fix_flow_{pump}', model._flow[model._link_index[pump]], Domain.equalsTo(0.0))
            model.constraint(f'fix_phi_pump_{pump}', model._phi_pump[model._set_of_pumps.index(pump)], Domain.equalsTo(0.0))
    for valve, status in gurobi_solution['valve_status'].items():
        if status == 0:
            model.constraint(f'fix_flow_{valve}', model._flow[model._link_index[valve]], Domain.equalsTo(0.0))

def MosekFixedOperationalModel(model_name, results_directory, junctions, tanks, pipes, pumps, valves, hydraulic_timestep, head_loss_equation, gurobi_solution):
    model = Model(model_name)
    reservoirs = {label: junction for label, junction in junctions.items() if junction.segment == 'reservoir'}
    model._junctions, model._reservoirs, model._tanks, model._pipes, model._pumps, model._valves = junctions, reservoirs, tanks, pipes, pumps, valves
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model, head_loss_equation)
    FixStatus(model, gurobi_solution)
    DefineIncidenceMatrix(model)
    DefineObjective(model, head_loss_equation)
    AddPowerConeConstraints(model, head_loss_equation)
    AddJunctionBalanceConstraints(model)

    #model.setSolverParam('ptfWriteSolutions', 'on')
    model.setSolverParam('numThreads', 12)

    model.acceptedSolutionStatus(AccSolutionStatus.Feasible)
    with open(os.path.join(results_directory, f'{model_name}_log.log'), 'w') as f:
        model.setLogHandler(f)
        model.solve()
    model.setLogHandler(None)
    #model.writeTask(os.path.join(results_directory, f'{model_name}_solution.ptf'))

    problem_status = 'feasible' if model.getPrimalSolutionStatus() in (SolutionStatus.Optimal, SolutionStatus.Feasible) else 'infeasible' if model.getProblemStatus() == ProblemStatus.PrimalInfeasible else 'nosolution'
    solution_time = model.getSolverDoubleInfo('optimizerTime')

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            solution['head'] = {junction: -cons.dual()[0] for junction, cons in model._balance_cons.items()} | {reservoir: model._reservoirs[reservoir].head_lower_bound[0] for reservoir in model._set_of_reservoirs} | {tank: model._tanks[tank].head_lower_bound[0] for tank in model._set_of_tanks}

        if len(model._set_of_links) > 0:
            flow_solution_levels = np.array(model._flow.level())
            solution['flow'] = {link: flow_solution_levels[i] for i, link in enumerate(model._set_of_links)}

            if len(model._set_of_tanks) > 0:
                solution['next_head'] = {tank: solution['head'][tank] + (hydraulic_timestep / model._tanks[tank].area) * float(np.dot(model._incidence[tank], flow_solution_levels)) for tank in model._set_of_tanks}

    model.dispose()
    return problem_status, solution, solution_time