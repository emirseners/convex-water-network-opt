import os
from gurobipy import GRB, Model, quicksum, nlfunc

def _variable_bounds(model):
    head_lower_bounds = {(node, t): model._nodes[node].head_lower_bound[t-1] for node in model._set_of_nodes for t in model._set_of_time_periods}
    head_upper_bounds = {(node, t): model._nodes[node].head_upper_bound[t-1] for node in model._set_of_nodes for t in model._set_of_time_periods}
    flow_lower_bounds = {(link, t): model._links[link].flow_lower_bound[t-1] for link in model._set_of_links for t in model._set_of_time_periods}
    flow_upper_bounds = {(link, t): model._links[link].flow_upper_bound[t-1] for link in model._set_of_links for t in model._set_of_time_periods}
    storage_lower_bounds = {(tank, t): model._tanks[tank].storage_initial if t == 0 else model._tanks[tank].storage_lower_bound for tank in model._set_of_tanks for t in model._set_of_time_periods_with_initial}
    storage_upper_bounds = {(tank, t): model._tanks[tank].storage_initial if t == 0 else model._tanks[tank].storage_upper_bound for tank in model._set_of_tanks for t in model._set_of_time_periods_with_initial}
    return head_lower_bounds, head_upper_bounds, flow_lower_bounds, flow_upper_bounds, storage_lower_bounds, storage_upper_bounds

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_non_reservoir_junctions = [label for label, junction in model._junctions.items() if junction.segment != 'reservoir']
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_time_periods = list(range(1, model._time_steps + 1))
    model._set_of_time_periods_with_initial = list(range(0, model._time_steps + 1))
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_links = list(model._links.keys())

def DefineDecisionVariables(model):
    head_lb, head_ub, flow_lb, flow_ub, storage_lb, storage_ub = _variable_bounds(model)

    model._head = model.addVars([(node, t) for node in model._set_of_nodes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_lb, ub=head_ub, name="head")
    model._flow = model.addVars([(link, t) for link in model._set_of_links for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=flow_lb, ub=flow_ub, name="flow")
    model._storage = model.addVars([(tank, t) for tank in model._set_of_tanks for t in model._set_of_time_periods_with_initial], vtype=GRB.CONTINUOUS, lb=storage_lb, ub=storage_ub, name="storage")
    model._pump_status = model.addVars([(pump, t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.BINARY, name="pump_status")
    model._valve_status = model.addVars([(valve, t) for valve in model._set_of_valves for t in model._set_of_time_periods], vtype=GRB.BINARY, name="valve_status")

def DefineObjective(model):
    model.setObjective(quicksum(model._pumps[pump].fixed_cost[t-1] * model._pump_status[pump, t] + model._pumps[pump].operational_cost[t-1] * model._flow[pump, t] for pump in model._set_of_pumps for t in model._set_of_time_periods), GRB.MINIMIZE)

def AddHeadLossConstraints(model, head_loss_equation):
    head_loss_lower_bounds = {(pipe, t): model._pipes[pipe].start_node.head_lower_bound[t-1] - model._pipes[pipe].end_node.head_upper_bound[t-1] for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_loss_upper_bounds = {(pipe, t): model._pipes[pipe].start_node.head_upper_bound[t-1] - model._pipes[pipe].end_node.head_lower_bound[t-1] for pipe in model._set_of_pipes for t in model._set_of_time_periods}
    head_loss = model.addVars([(pipe, t) for pipe in model._set_of_pipes for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_loss_lower_bounds, ub=head_loss_upper_bounds, name="head_loss")
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for pipe in model._set_of_pipes:
        for t in model._set_of_time_periods:
            model.addGenConstrNL(head_loss[pipe, t], model._pipes[pipe].loss_coefficient * nlfunc.signpow(model._flow[pipe, t], exponent=exponent), name=f'head_loss_def({pipe},{t})')
            model.addConstr(model._head[model._pipes[pipe].start_node.label, t] - model._head[model._pipes[pipe].end_node.label, t] == head_loss[pipe,t], name = f'head_loss({pipe},{t})')

def AddPumpFlowConstraints(model):
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            model.addConstr(model._flow[pump, t] - (model._pump_status[pump, t] * model._pumps[pump].flow_lower_bound[t-1]) >= 0, name = f'pump_flow_l({pump},{t})')
            model.addConstr((model._pump_status[pump, t] * model._pumps[pump].flow_upper_bound[t-1]) - model._flow[pump, t] >= 0, name = f'pump_flow_u({pump},{t})')

def AddValveFlowConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            model.addConstr(model._flow[valve, t] - (model._valve_status[valve, t] * model._valves[valve].flow_lower_bound[t-1]) >= 0, name = f'valve_flow_l({valve},{t})')
            model.addConstr((model._valve_status[valve, t] * model._valves[valve].flow_upper_bound[t-1]) - model._flow[valve, t] >= 0, name = f'valve_flow_u({valve},{t})')

def AddPumpHeadConstraints(model):
    head_gain_flow_lower_bounds = {(pump, t): -model._pumps[pump].curve_a for pump in model._set_of_pumps for t in model._set_of_time_periods}
    head_gain_flow = model.addVars([(pump, t) for pump in model._set_of_pumps for t in model._set_of_time_periods], vtype=GRB.CONTINUOUS, lb=head_gain_flow_lower_bounds, ub=0, name="head_gain_flow")
    for pump in model._set_of_pumps:
        for t in model._set_of_time_periods:
            model.addConstr(head_gain_flow[pump,t] == (model._pumps[pump].curve_b * (model._flow[pump,t] ** model._pumps[pump].curve_c)), name = f'pump_head_gain_flow({pump},{t})')
            model.addConstr((model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]) <= model._head[model._pumps[pump].start_node.label, t] - model._head[model._pumps[pump].end_node.label, t] + (model._pumps[pump].curve_a * model._pump_status[pump, t]) + head_gain_flow[pump,t], name = f'head_diff_lower({pump},{t})')
            model.addConstr((model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]) >= model._head[model._pumps[pump].start_node.label, t] - model._head[model._pumps[pump].end_node.label, t] + (model._pumps[pump].curve_a * model._pump_status[pump, t]) + head_gain_flow[pump,t], name = f'head_diff_upper({pump},{t})')

def AddValveHeadConstraints(model):
    for valve in model._set_of_valves:
        for t in model._set_of_time_periods:
            model.addConstr(model._head[model._valves[valve].start_node.label, t] - model._head[model._valves[valve].end_node.label, t] - ((model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve, t])) >= 0, name = f'valve_head_l({valve},{t})')
            model.addConstr(((model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve, t])) - model._head[model._valves[valve].start_node.label, t] + model._head[model._valves[valve].end_node.label, t] >= 0, name = f'valve_head_u({valve},{t})')

def AddTankBalanceConstraints(model, hydraulic_timestep): #m^3 / s == m^3
    for tank in model._set_of_tanks:
        for t in model._set_of_time_periods:
            model.addConstr(hydraulic_timestep * quicksum(model._flow[edge.label, t] for edge in model._tanks[tank].incoming_edges) - hydraulic_timestep * quicksum(model._flow[edge.label, t] for edge in model._tanks[tank].outgoing_edges) - model._storage[tank, t] + model._storage[tank, t-1] == 0, name = f'tank_balance({tank},{t})')

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    for junction in model._set_of_non_reservoir_junctions:
        for t in model._set_of_time_periods:
            model.addConstr(quicksum(model._flow[edge.label, t] for edge in model._junctions[junction].incoming_edges) - quicksum(model._flow[edge.label, t] for edge in model._junctions[junction].outgoing_edges) - model._junctions[junction].net_demand[t-1] == 0, name = f'junction_balance({junction},{t})')

def AddTankHeadConstraints(model):
    for tank in model._set_of_tanks:
        for t in model._set_of_time_periods:
            model.addConstr(model._head[tank, t] == model._tanks[tank].elevation + ((1 / model._tanks[tank].area) * model._storage[tank, t]), name = f'tank_head({tank},{t})')

def GurobiOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    model = Model(epanet_model)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model, head_loss_equation)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
    AddPumpHeadConstraints(model)
    AddValveHeadConstraints(model)
    AddTankBalanceConstraints(model, hydraulic_timestep)
    AddJunctionBalanceConstraints(model)
    AddTankHeadConstraints(model)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    #lp_filename = os.path.join(results_dir, f'{epanet_model}.lp')
    #model.write(lp_filename)
    model.setParam('TimeLimit', 3600)
    log_file_path = os.path.join(results_dir, epanet_model + 'GurobiLog.txt')
    model.setParam('LogFile', log_file_path)

    model.optimize()