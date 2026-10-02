import os
import numpy as np
from gurobipy import GRB, Model, Env, quicksum, nlfunc

def _variable_bounds(model):
    head_lower_bounds = {(node,t): model._nodes[node].head_lower_bound[t-1] for (node, t) in model._head_indices}
    head_upper_bounds = {(node,t): model._nodes[node].head_upper_bound[t-1] for (node, t) in model._head_indices}
    flow_lower_bounds = {(link,t): model._links[link].flow_lower_bound[t-1] for link in model._set_of_links for t in model._set_of_time_periods}
    flow_upper_bounds = {(link,t): model._links[link].flow_upper_bound[t-1] for link in model._set_of_links for t in model._set_of_time_periods}
    return head_lower_bounds, head_upper_bounds, flow_lower_bounds, flow_upper_bounds

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

def DefineDecisionVariables(model):
    model._head_indices = ([(node,t) for node in model._set_of_nodes for t in model._set_of_time_periods] + [(tank, model._time_steps + 1) for tank in model._set_of_tanks])
    head_lb, head_ub, flow_lb, flow_ub = _variable_bounds(model)

    model._head = model.addVars([(node,t) for (node,t) in model._head_indices], vtype=GRB.CONTINUOUS, lb=head_lb, ub=head_ub, name="head")
    model._flow = model.addVars([(link,t) for link in model._set_of_links for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=flow_lb, ub=flow_ub, name="flow")
    model._pump_status = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.BINARY, name="pump_status")
    model._valve_status = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.BINARY, name="valve_status")

def DefineObjective(model):
    model.setObjective(quicksum(model._pumps[pump].fixed_cost[t-1] * model._pump_status[pump,t] + model._pumps[pump].operational_cost[t-1] * model._flow[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods), GRB.MINIMIZE)

def AddHeadLossConstraints(model):
    head_loss_lower_bounds = {(pipe,t): model._pipes[pipe].start_node.head_lower_bound[t-1] - model._pipes[pipe].end_node.head_upper_bound[t-1] for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_loss_upper_bounds = {(pipe,t): model._pipes[pipe].start_node.head_upper_bound[t-1] - model._pipes[pipe].end_node.head_lower_bound[t-1] for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_loss = model.addVars([(pipe,t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_loss_lower_bounds, ub=head_loss_upper_bounds, name="head_loss")
    for pipe in model._set_of_pipes:
        for t in model._set_of_time_periods:
            if model._pipes[pipe].flow_lower_bound[t-1] == model._pipes[pipe].flow_upper_bound[t-1]:
                model.addConstr(model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == model._pipes[pipe].loss_coefficient * (-1 if model._pipes[pipe].flow_lower_bound[t-1] < 0.0 else 1) * (abs(model._pipes[pipe].flow_lower_bound[t-1]) ** model._pipes[pipe].exponent), name=f'head_loss({pipe},{t})')
                head_loss[pipe,t].lb = head_loss[pipe,t].ub = model._pipes[pipe].loss_coefficient * (-1 if model._pipes[pipe].flow_lower_bound[t-1] < 0.0 else 1) * (abs(model._pipes[pipe].flow_lower_bound[t-1]) ** model._pipes[pipe].exponent)
            else:
                model.addGenConstrNL(head_loss[pipe,t], model._pipes[pipe].loss_coefficient * nlfunc.signpow(model._flow[pipe,t], exponent=model._pipes[pipe].exponent), name=f'head_loss_def({pipe},{t})')
                model.addConstr(model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == head_loss[pipe,t], name=f'head_loss({pipe},{t})')

def AddPumpFlowConstraints(model):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].decisions[t-1] == 1:
                model._pump_status[pump,t].lb = model._pump_status[pump,t].ub = 1
            elif model._pumps[pump].decisions[t-1] == 0:
                model._flow[pump,t].lb = model._flow[pump,t].ub = 0.0
                model._pump_status[pump,t].lb = model._pump_status[pump,t].ub = 0
            else:
                model.addConstr(model._flow[pump,t] - (model._pump_status[pump,t] * model._pumps[pump].flow_lower_bound[t-1]) >= 0, name=f'pump_flow_l({pump},{t})')
                model.addConstr((model._pump_status[pump,t] * model._pumps[pump].flow_upper_bound[t-1]) - model._flow[pump,t] >= 0, name=f'pump_flow_u({pump},{t})')

def AddValveFlowConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].decisions[t-1] == 1:
                model._valve_status[valve,t].lb = model._valve_status[valve,t].ub = 1
            elif model._valves[valve].decisions[t-1] == 0:
                model._valve_status[valve,t].lb = model._valve_status[valve,t].ub = 0
                model._flow[valve,t].lb = model._flow[valve,t].ub = 0.0
            else:
                model.addConstr(model._flow[valve,t] - (model._valve_status[valve,t] * model._valves[valve].flow_lower_bound[t-1]) >= 0, name=f'valve_flow_l({valve},{t})')
                model.addConstr((model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1]) - model._flow[valve,t] >= 0, name=f'valve_flow_u({valve},{t})')

def AddPumpHeadConstraints(model):
    head_gain_flow_lower_bounds = {(pump,t): model._pumps[pump].curve_b * (model._pumps[pump].flow_upper_bound[t-1] ** model._pumps[pump].curve_c) for pump in model._set_of_pumps for t in model._set_of_time_periods}
    head_gain_flow = model.addVars([(pump,t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_gain_flow_lower_bounds, ub=0.0, name="head_gain_flow")
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            if model._pumps[pump].decisions[t-1] == 0:
                head_gain_flow[pump,t].lb = 0.0
            elif model._pumps[pump].flow_lower_bound[t-1] == model._pumps[pump].flow_upper_bound[t-1] and model._pumps[pump].flow_lower_bound[t-1] > 0.0:
                if model._pumps[pump].curve_c == 1.0:
                    model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1]) == 0, name=f'head_diff_eq({pump},{t})')
                    head_gain_flow[pump,t].lb = head_gain_flow[pump,t].ub = model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1])
                else:
                    model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c) == 0, name=f'head_diff_eq({pump},{t})')
                    head_gain_flow[pump,t].lb = head_gain_flow[pump,t].ub = model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c)
            elif model._pumps[pump].decisions[t-1] == 1:
                if model._pumps[pump].curve_c == 1.0:
                    model.addConstr(head_gain_flow[pump,t] == model._pumps[pump].curve_b * model._flow[pump,t], name=f'pump_head_gain_flow({pump},{t})')
                    model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + head_gain_flow[pump,t] == 0, name=f'head_diff_eq({pump},{t})')
                    head_gain_flow[pump,t].ub = model._pumps[pump].curve_b * model._pumps[pump].flow_lower_bound[t-1]
                else:
                    model.addGenConstrNL(head_gain_flow[pump,t], model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c), name=f'pump_head_gain_flow({pump},{t})')
                    model.addConstr(model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + model._pumps[pump].curve_a + head_gain_flow[pump,t] == 0, name=f'head_diff_eq({pump},{t})')
                    head_gain_flow[pump,t].ub = model._pumps[pump].curve_b * (model._pumps[pump].flow_lower_bound[t-1] ** model._pumps[pump].curve_c)
            else:
                if model._pumps[pump].curve_c == 1.0:
                    model.addConstr(head_gain_flow[pump,t] == model._pumps[pump].curve_b * model._flow[pump,t], name=f'pump_head_gain_flow({pump},{t})')
                    model.addConstr((model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]) <= model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + head_gain_flow[pump,t], name=f'head_diff_lower({pump},{t})')
                    model.addConstr((model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]) >= model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + head_gain_flow[pump,t], name=f'head_diff_upper({pump},{t})')
                else:
                    model.addGenConstrNL(head_gain_flow[pump,t], model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c), name=f'pump_head_gain_flow({pump},{t})')
                    model.addConstr((model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]) <= model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + head_gain_flow[pump,t], name=f'head_diff_lower({pump},{t})')
                    model.addConstr((model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]) >= model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + head_gain_flow[pump,t], name=f'head_diff_upper({pump},{t})')

def AddValveHeadConstraints(model):
    head_loss_upper_bounds = {(valve,t): model._valves[valve].loss_coefficient * (model._valves[valve].flow_upper_bound[t-1] ** model._valves[valve].exponent) for valve in model._set_of_valves for t in model._set_of_time_periods}
    valve_head_loss = model.addVars([(valve,t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=0.0, ub=head_loss_upper_bounds, name="valve_head_loss")

    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            if model._valves[valve].decisions[t-1] == 0:
                valve_head_loss[valve,t].lb = valve_head_loss[valve,t].ub = 0.0
                continue
            if model._valves[valve].flow_lower_bound[t-1] == model._valves[valve].flow_upper_bound[t-1]:
                valve_head_loss[valve,t].lb = valve_head_loss[valve,t].ub = model._valves[valve].loss_coefficient * (model._valves[valve].flow_lower_bound[t-1] ** model._valves[valve].exponent)
            else:
                model.addGenConstrNL(valve_head_loss[valve,t], model._valves[valve].loss_coefficient * nlfunc.signpow(model._flow[valve,t], exponent=model._valves[valve].exponent), name=f'valve_head_loss_def({valve},{t})')
            if model._valves[valve].decisions[t-1] == 1:
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - valve_head_loss[valve,t] == 0, name=f'valve_head_loss({valve},{t})')
            else:
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - valve_head_loss[valve,t] >= (model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve,t]), name=f'valve_head_loss_l({valve},{t})')
                model.addConstr(model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] - valve_head_loss[valve,t] <= (model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve,t]), name=f'valve_head_loss_u({valve},{t})')

def AddTankBalanceConstraints(model, hydraulic_timestep):
    for tank in model._set_of_tanks:
        for t in model._set_of_time_periods:
            model.addConstr((hydraulic_timestep / model._tanks[tank].area) * quicksum(model._flow[edge.label,t] for edge in model._tanks[tank].incoming_edges) - (hydraulic_timestep / model._tanks[tank].area) * quicksum(model._flow[edge.label,t] for edge in model._tanks[tank].outgoing_edges) == model._head[tank,t+1] - model._head[tank,t], name=f'tank_balance({tank},{t})')

def AddJunctionBalanceConstraints(model):
    for junction in model._set_of_non_reservoir_junctions:
        for t in model._set_of_time_periods:
            model.addConstr(quicksum(model._flow[edge.label,t] for edge in model._junctions[junction].incoming_edges) - quicksum(model._flow[edge.label,t] for edge in model._junctions[junction].outgoing_edges) == model._junctions[junction].net_demand[t-1], name=f'junction_balance({junction},{t})')

def GurobiOperationalModel(model_name, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit, test_solution=None, test_continuous=True):
    model_env = Env(empty=True)
    model_env.start()
    model = Model(model_name, env=model_env)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
    AddPumpHeadConstraints(model)
    AddValveHeadConstraints(model)
    AddTankBalanceConstraints(model, hydraulic_timestep)
    AddJunctionBalanceConstraints(model)

    if test_solution is not None:
        vars_to_fix = ([model._pump_status[pump,t] for pump in model._set_of_pumps for t in model._set_of_time_periods] + [model._valve_status[valve,t] for valve in model._set_of_valves for t in model._set_of_time_periods])
        vals_to_fix = ([test_solution['pump_status'][pump][t-1] for pump in model._set_of_pumps for t in model._set_of_time_periods] + [test_solution['valve_status'][valve][t-1] for valve in model._set_of_valves for t in model._set_of_time_periods])

        if test_continuous:
            vars_to_fix += ([model._head[node,t] for (node,t) in model._head_indices] + [model._flow[link,t] for link in model._set_of_links for t in model._set_of_time_periods])
            vals_to_fix += ([test_solution['head'][node][t-1] for (node,t) in model._head_indices] + [test_solution['flow'][link][t-1] for link in model._set_of_links for t in model._set_of_time_periods])

        model.setAttr("LB", vars_to_fix, vals_to_fix)
        model.setAttr("UB", vars_to_fix, vals_to_fix)
        model.update()

    #lp_filename=os.path.join(results_directory, f'{model_name}.lp')
    #model.write(lp_filename)
    model.setParam('TimeLimit', time_limit)
    #model.setParam('MIPFocus', 3)
    model.setParam('Threads', 12)
    log_file_path = os.path.join(results_directory, f'{model_name}_log.txt')
    model.setParam('LogFile', log_file_path)

    model.optimize()

    problem_status = 'feasible' if model.SolCount > 0 else 'infeasible' if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD) else 'nosolution'

    lower_bound = model.ObjBound if problem_status != 'infeasible' and abs(model.ObjBound) < GRB.INFINITY else None
    upper_bound = model.ObjVal if problem_status == 'feasible' else None
    solution_time = model.Runtime

    if problem_status == 'infeasible':
        model.setParam('TimeLimit', 60)
        model.computeIIS()
        iis_file_path = os.path.join(results_directory, f'IIS_{model_name}.ilp')
        model.write(iis_file_path)

    #model.write(os.path.join(results_directory, f'{model_name}_solution.sol'))

    solution = {} if problem_status == 'feasible' else None
    if problem_status == 'feasible':
        if len(model._set_of_nodes) > 0:
            solution['head'] = {node: np.array([model._head[node, t].x for t in (model._set_of_time_periods if node in model._set_of_junctions else model._set_of_time_periods_with_terminal)]) for node in model._set_of_nodes}

        if len(model._set_of_links) > 0:
            solution['flow'] = {link: np.array([model._flow[link, t].x for t in model._set_of_time_periods]) for link in model._set_of_links}

        if len(model._set_of_pumps) > 0:
            solution['pump_status'] = {pump: np.rint([model._pump_status[pump, t].x for t in model._set_of_time_periods]) for pump in model._set_of_pumps}

        if len(model._set_of_valves) > 0:
            solution['valve_status'] = {valve: np.rint([model._valve_status[valve, t].x for t in model._set_of_time_periods]) for valve in model._set_of_valves}

    model.dispose()
    model_env.dispose()

    return problem_status, solution, lower_bound, upper_bound, solution_time