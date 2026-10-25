import os
import numpy as np
from gurobipy import GRB, Model, Env, quicksum, nlfunc

def _variable_bounds(model):
    head_lower_bounds = {(node,t): model._nodes[node].head_lower_bound[t-1] for (node,t) in model._head_indices}
    head_upper_bounds = {(node,t): model._nodes[node].head_upper_bound[t-1] for (node,t) in model._head_indices}
    flow_lower_bounds = {(link,t): model._links[link].flow_lower_bound[t-1] if link not in model._set_of_pipe_families else min((pipe.flow_lower_bound[t-1] for pipe in model._pipe_families[link].options if pipe.selection_decision != 0), default=0.0) for link in model._set_of_links for t in model._set_of_time_periods}
    flow_upper_bounds = {(link,t): model._links[link].flow_upper_bound[t-1] if link not in model._set_of_pipe_families else max((pipe.flow_upper_bound[t-1] for pipe in model._pipe_families[link].options if pipe.selection_decision != 0), default=0.0) for link in model._set_of_links for t in model._set_of_time_periods}
    pipe_head_loss_lower_bounds = {(pipe_family,t): min(0.0, model._pipe_families[pipe_family].start_node.head_lower_bound[t-1] - model._pipe_families[pipe_family].end_node.head_upper_bound[t-1]) for pipe_family in model._set_of_pipe_families for t in model._set_of_time_periods}
    pipe_head_loss_upper_bounds = {(pipe_family,t): max(0.0, model._pipe_families[pipe_family].start_node.head_upper_bound[t-1] - model._pipe_families[pipe_family].end_node.head_lower_bound[t-1]) for pipe_family in model._set_of_pipe_families for t in model._set_of_time_periods}
    valve_head_loss_upper_bounds = {(valve,t): model._valves[valve].resistance_coefficient * (model._valves[valve].flow_upper_bound[t-1] ** model._valves[valve].exponent) for valve in model._set_of_valves for t in model._set_of_time_periods}
    pump_head_gain_lower_bounds = {(pump,t): model._pumps[pump].curve_b * (model._pumps[pump].flow_upper_bound[t-1] ** model._pumps[pump].curve_c) for pump in model._set_of_pumps for t in model._set_of_time_periods}
    pump_head_gain_upper_bounds = {(pump,t): model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c) for pump in model._set_of_pumps for t in model._set_of_time_periods}
    return head_lower_bounds, head_upper_bounds, flow_lower_bounds, flow_upper_bounds, pipe_head_loss_lower_bounds, pipe_head_loss_upper_bounds, valve_head_loss_upper_bounds, pump_head_gain_lower_bounds, pump_head_gain_upper_bounds

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_non_reservoir_junctions = [label for label, junction in model._junctions.items() if junction.segment != 'reservoir']
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_time_periods = list(range(1, model._time_steps + 1))
    model._set_of_time_periods_with_terminal = list(range(1, model._time_steps + 2))
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_links = list(model._links.keys())
    model._set_of_pipe_families = list(model._pipe_families.keys())
    model._incoming_edges = {node: [] for node in model._set_of_nodes}
    model._outgoing_edges = {node: [] for node in model._set_of_nodes}

    for label, link in model._links.items():
        model._incoming_edges[link.end_node.label].append(label)
        model._outgoing_edges[link.start_node.label].append(label)

def DefineDecisionVariables(model):
    model._head_indices = ([(node,t) for node in model._set_of_nodes for t in model._set_of_time_periods] + [(tank, model._time_steps + 1) for tank in model._set_of_tanks])
    head_lb, head_ub, flow_lb, flow_ub, pipe_head_loss_lb, pipe_head_loss_ub, valve_head_loss_ub, pump_head_gain_lb, pump_head_gain_ub = _variable_bounds(model)

    model._head = model.addVars(model._head_indices, vtype=GRB.CONTINUOUS, lb=head_lb, ub=head_ub, name="head")
    model._flow = model.addVars([(link,t) for link in model._set_of_links for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=flow_lb, ub=flow_ub, name="flow")
    model._pump_status = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.BINARY, lb={(pump,t): 1 if model._pumps[pump].status_decisions[t-1] == 1 else 0 for pump in model._set_of_pumps for t in model._set_of_time_periods}, ub={(pump,t): 0 if model._pumps[pump].status_decisions[t-1] == 0 else 1 for pump in model._set_of_pumps for t in model._set_of_time_periods}, name="pump_status")
    model._valve_status = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.BINARY, lb={(valve,t): 1 if model._valves[valve].status_decisions[t-1] == 1 else 0 for valve in model._set_of_valves for t in model._set_of_time_periods}, ub={(valve,t): 0 if model._valves[valve].status_decisions[t-1] == 0 else 1 for valve in model._set_of_valves for t in model._set_of_time_periods}, name="valve_status")

    if model._problem != 'operational':
        model._pipe_status = model.addVars(model._set_of_pipes, vtype=GRB.BINARY, lb={pipe: 1 if model._pipes[pipe].selection_decision == 1 else 0 for pipe in model._set_of_pipes}, ub={pipe: 0 if model._pipes[pipe].selection_decision == 0 else 1 for pipe in model._set_of_pipes}, name="pipe_status")
    else:
        model._pipe_status = {pipe: 1.0 for pipe in model._set_of_pipes}

    model._pipe_head_loss = model.addVars([(pipe_family,t) for pipe_family in model._set_of_pipe_families for t in model._set_of_time_periods if not all(pipe.fixed_flow[t-1] for pipe in model._pipe_families[pipe_family].options)], vtype=GRB.CONTINUOUS, lb=pipe_head_loss_lb, ub=pipe_head_loss_ub, name="head_loss")
    model._valve_head_loss = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods if model._valves[valve].status_decisions[t-1] != 0 and not model._valves[valve].fixed_flow[t-1]], vtype=GRB.CONTINUOUS, lb=0.0, ub=valve_head_loss_ub, name="valve_head_loss")
    model._pump_head_gain = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods if model._pumps[pump].status_decisions[t-1] != 0 and model._pumps[pump].curve_c != 1.0 and not model._pumps[pump].fixed_flow[t-1]], vtype=GRB.CONTINUOUS, lb=pump_head_gain_lb, ub=pump_head_gain_ub, name="pump_head_gain")

def DefineObjective(model):
    model.setObjective(quicksum(model._pumps[pump].fixed_cost[t-1] * model._pump_status[pump,t] + model._pumps[pump].operational_cost[t-1] * model._flow[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods) + quicksum(model._pipes[pipe].installation_cost * model._pipe_status[pipe] for pipe in model._set_of_pipes if model._pipes[pipe].installation_cost), GRB.MINIMIZE)

def AddHeadLossConstraints(model):
    for pipe_family in model._set_of_pipe_families:
        for t in model._set_of_time_periods:
            if all(pipe.fixed_flow[t-1] for pipe in model._pipe_families[pipe_family].options):
                head_loss = sum(pipe.resistance_coefficient * (-1 if pipe.flow_lower_bound[t-1] < 0.0 else 1) * (abs(pipe.flow_lower_bound[t-1]) ** pipe.exponent) for pipe in model._pipe_families[pipe_family].options)
            else:
                model.addGenConstrNL(model._pipe_head_loss[pipe_family,t], quicksum(pipe.resistance_coefficient * model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) * nlfunc.signpow(model._flow[pipe_family,t], exponent=model._pipe_families[pipe_family].exponent), name=f'head_loss_def({pipe_family},{t})')
                head_loss = model._pipe_head_loss[pipe_family,t]

            if not model._pipe_families[pipe_family].is_optional or any(pipe.selection_decision == 1 for pipe in model._pipe_families[pipe_family].options):
                model.addConstr(model._head[model._pipe_families[pipe_family].start_node.label,t] - model._head[model._pipe_families[pipe_family].end_node.label,t] - head_loss == 0, name=f'head_loss({pipe_family},{t})')
            else:
                model.addConstr(model._head[model._pipe_families[pipe_family].start_node.label,t] - model._head[model._pipe_families[pipe_family].end_node.label,t] - head_loss >= (model._pipe_families[pipe_family].start_node.head_lower_bound[t-1] - model._pipe_families[pipe_family].end_node.head_upper_bound[t-1]) * (1 - quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options)), name=f'head_loss_l({pipe_family},{t})')
                model.addConstr(model._head[model._pipe_families[pipe_family].start_node.label,t] - model._head[model._pipe_families[pipe_family].end_node.label,t] - head_loss <= (model._pipe_families[pipe_family].start_node.head_upper_bound[t-1] - model._pipe_families[pipe_family].end_node.head_lower_bound[t-1]) * (1 - quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options)), name=f'head_loss_u({pipe_family},{t})')

def AddPipeFlowConstraints(model):
    if model._problem == 'operational': return
    for pipe_family in model._set_of_pipe_families:
        if model._pipe_families[pipe_family].is_optional:
            model.addConstr(quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) <= 1, name=f'pipe_family_binary({pipe_family})')
        else:
            model.addConstr(quicksum(model._pipe_status[pipe.label] for pipe in model._pipe_families[pipe_family].options) == 1, name=f'pipe_family_binary({pipe_family})')

        for t in model._set_of_time_periods:
            model.addConstr(model._flow[pipe_family,t] - quicksum(model._pipe_status[pipe.label] * pipe.flow_lower_bound[t-1] for pipe in model._pipe_families[pipe_family].options) >= 0, name=f'pipe_family_flow_l({pipe_family},{t})')
            model.addConstr(quicksum(model._pipe_status[pipe.label] * pipe.flow_upper_bound[t-1] for pipe in model._pipe_families[pipe_family].options) - model._flow[pipe_family,t] >= 0, name=f'pipe_family_flow_u({pipe_family},{t})')

def AddPumpFlowConstraints(model):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].status_decisions[t-1] == -1:
                model.addConstr(model._flow[pump,t] - model._pump_status[pump,t] * model._pumps[pump].positive_flow_lower_bound[t-1] >= 0, name=f'pump_flow_l({pump},{t})')
                model.addConstr((model._pump_status[pump,t] * model._pumps[pump].flow_upper_bound[t-1]) - model._flow[pump,t] >= 0, name=f'pump_flow_u({pump},{t})')

def AddValveFlowConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].status_decisions[t-1] == -1:
                model.addConstr(model._flow[valve,t] - model._valve_status[valve,t] * model._valves[valve].positive_flow_lower_bound[t-1] >= 0, name=f'valve_flow_l({valve},{t})')
                model.addConstr((model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1]) - model._flow[valve,t] >= 0, name=f'valve_flow_u({valve},{t})')

def AddPumpHeadConstraints(model):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].status_decisions[t-1] == 0: continue
            if model._pumps[pump].fixed_flow[t-1]:
                pump_head_gain = model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c)
            elif model._pumps[pump].curve_c == 1.0:
                pump_head_gain = model._pumps[pump].curve_b * model._flow[pump,t]
            else:
                model.addGenConstrNL(model._pump_head_gain[pump,t], model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c), name=f'pump_pump_head_gain({pump},{t})')
                pump_head_gain = model._pump_head_gain[pump,t]
            if model._pumps[pump].status_decisions[t-1] == 1:
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + pump_head_gain == 0, name=f'head_diff_eq({pump},{t})')
            else:
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] + pump_head_gain >= (model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]), name=f'head_diff_lower({pump},{t})')
                model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a * model._pump_status[pump,t] + pump_head_gain <= (model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]), name=f'head_diff_upper({pump},{t})')

def AddValveHeadConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].status_decisions[t-1] == 0: continue

            if model._valves[valve].fixed_flow[t-1]:
                head_loss = model._valves[valve].resistance_coefficient * (model._valves[valve].flow_lower_bound[t-1] ** model._valves[valve].exponent)
            else:
                model.addGenConstrNL(model._valve_head_loss[valve,t], model._valves[valve].resistance_coefficient * (model._flow[valve,t] ** model._valves[valve].exponent), name=f'valve_head_loss_def({valve},{t})')
                head_loss = model._valve_head_loss[valve,t]

            if model._valves[valve].status_decisions[t-1] == 1:
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - head_loss == 0, name=f'valve_head_loss({valve},{t})')
            else:
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - head_loss >= (model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve,t]), name=f'valve_head_loss_l({valve},{t})')
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - head_loss <= (model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve,t]), name=f'valve_head_loss_u({valve},{t})')

def AddTankBalanceConstraints(model):
    for tank in model._set_of_tanks:
        for t in model._set_of_time_periods:
            model.addConstr(model._tanks[tank].tank_coef * (quicksum(model._flow[link,t] for link in model._incoming_edges[tank]) - quicksum(model._flow[link,t] for link in model._outgoing_edges[tank])) == model._head[tank,t+1] - model._head[tank,t], name=f'tank_balance({tank},{t})')

def AddJunctionBalanceConstraints(model):
    for junction in model._set_of_non_reservoir_junctions:
        for t in model._set_of_time_periods:
            model.addConstr(quicksum(model._flow[link,t] for link in model._incoming_edges[junction]) - quicksum(model._flow[link,t] for link in model._outgoing_edges[junction]) == model._junctions[junction].net_demand[t-1], name=f'junction_balance({junction},{t})')

def GurobiModel(model_name, network, test_solution=None, test_continuous=True):
    model_env = Env(empty=True)
    model_env.start()
    model = Model(model_name, env=model_env)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps, model._problem = network.junctions, network.tanks, network.pipes, network.pumps, network.valves, network.time_steps, network.problem
    model._nodes, model._links, model._pipe_families = network.junctions | network.tanks, network.pipe_families | network.pumps | network.valves, network.pipe_families

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model)
    AddPipeFlowConstraints(model)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
    AddPumpHeadConstraints(model)
    AddValveHeadConstraints(model)
    AddTankBalanceConstraints(model)
    AddJunctionBalanceConstraints(model)

    if test_solution is not None:
        vars_to_fix = ([model._pump_status[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods] + [model._valve_status[valve,t] for valve in model._set_of_valves for t in model._set_of_time_periods])
        vals_to_fix = ([test_solution['pump_status'][pump][t-1] for pump in model._set_of_pumps for t in model._set_of_time_periods] + [test_solution['valve_status'][valve][t-1] for valve in model._set_of_valves for t in model._set_of_time_periods])

        if model._problem != 'operational':
            vars_to_fix += [model._pipe_status[pipe] for pipe in model._set_of_pipes]
            vals_to_fix += [test_solution['pipe_status'][pipe] for pipe in model._set_of_pipes]

        if test_continuous:
            vars_to_fix += ([model._head[node,t] for (node,t) in model._head_indices] + [model._flow[link,t] for link in model._set_of_links for t in model._set_of_time_periods])
            vals_to_fix += ([test_solution['head'][node][t-1] for (node,t) in model._head_indices] + [test_solution['flow'][link][t-1] for link in model._set_of_links for t in model._set_of_time_periods])

        model.setAttr("LB", vars_to_fix, vals_to_fix)
        model.setAttr("UB", vars_to_fix, vals_to_fix)
        model.update()

    #lp_filename=os.path.join(results_directory, f'{model_name}.lp')
    #model.write(lp_filename)
    model.setParam('TimeLimit', network.time_limit)
    model.setParam('MIPGap', network.optimality_gap)
    #model.setParam('MIPFocus', 3)
    model.setParam('Threads', 12)
    log_file_path = os.path.join(network.results_directory, f'{model_name}_log.txt')
    model.setParam('LogFile', log_file_path)

    model.optimize()

    problem_status = 'feasible' if model.SolCount > 0 else 'infeasible' if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else 'nosolution'

    lower_bound = model.ObjBound if problem_status != 'infeasible' and abs(model.ObjBound) < GRB.INFINITY else None
    upper_bound = model.ObjVal if problem_status == 'feasible' else None
    solution_time = model.Runtime

    if problem_status == 'infeasible':
        model.setParam('TimeLimit', 60)
        model.computeIIS()
        iis_file_path = os.path.join(network.results_directory, f'IIS_{model_name}.ilp')
        model.write(iis_file_path)

    #model.write(os.path.join(results_directory, f'{model_name}_solution.sol'))

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            solution['head'] = {node: np.array([model._head[node,t].x for t in (model._set_of_time_periods if node in model._set_of_junctions else model._set_of_time_periods_with_terminal)]) for node in model._set_of_nodes}

        if len(model._set_of_links) > 0:
            solution['flow'] = {link: np.array([model._flow[link,t].x for t in model._set_of_time_periods]) for link in model._set_of_links}

        if len(model._set_of_pumps) > 0:
            solution['pump_status'] = {pump: np.rint([model._pump_status[pump,t].x for t in model._set_of_time_periods]) for pump in model._set_of_pumps}

        if len(model._set_of_valves) > 0:
            solution['valve_status'] = {valve: np.rint([model._valve_status[valve,t].x for t in model._set_of_time_periods]) for valve in model._set_of_valves}

        if len(model._set_of_pipes) > 0:
            solution['pipe_status'] = {pipe: int(round(model._pipe_status[pipe].x)) if model._problem != 'operational' else 1 for pipe in model._set_of_pipes}

    model.dispose()
    model_env.dispose()

    return problem_status, solution, lower_bound, upper_bound, solution_time