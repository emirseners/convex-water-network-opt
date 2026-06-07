import os
import tempfile
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager

def _headbounds(model, i, t):
    return model._nodes[i].head_lower_bound[t-1], model._nodes[i].head_upper_bound[t-1]

def _flowbounds(model, i, t):
    return model._links[i].flow_lower_bound[t-1], model._links[i].flow_upper_bound[t-1]

def _storage_bounds(model, i, t):
    if t == 0:
        return model._tanks[i].storage_initial, model._tanks[i].storage_initial
    else:
        return model._tanks[i].storage_lower_bound, model._tanks[i].storage_upper_bound

def _objective_function(model):
    return pyo.quicksum((model._pumps[i].fixed_cost[t-1] * model._pump_status[i, t]) + (model._pumps[i].operational_cost[t-1] * model._flow[i, t]) for i in model._set_of_pumps for t in model._set_of_time_periods)

def DefineSets(model):
    model._set_of_junctions = pyo.Set(initialize=list(model._junctions.keys()))
    model._set_of_non_reservoir_junctions = pyo.Set(initialize=[label for label, junction in model._junctions.items() if junction.segment != 'reservoir'])
    model._set_of_tanks = pyo.Set(initialize=list(model._tanks.keys()))
    model._set_of_pipes = pyo.Set(initialize=list(model._pipes.keys()))
    model._set_of_pumps = pyo.Set(initialize=list(model._pumps.keys()))
    model._set_of_valves = pyo.Set(initialize=list(model._valves.keys()))
    model._set_of_time_periods = pyo.RangeSet(1, model._time_steps)
    model._set_of_time_periods_with_initial = pyo.RangeSet(0, model._time_steps)
    model._set_of_nodes = pyo.Set(initialize=list(model._nodes.keys()))
    model._set_of_links = pyo.Set(initialize=list(model._links.keys()))

def DefineDecisionVariables(model):
    model._head = pyo.Var(model._set_of_nodes, model._set_of_time_periods, bounds=_headbounds)
    model._flow = pyo.Var(model._set_of_links, model._set_of_time_periods, bounds=_flowbounds)
    model._storage = pyo.Var(model._set_of_tanks, model._set_of_time_periods_with_initial, bounds=_storage_bounds)
    model._pump_status = pyo.Var(model._set_of_pumps, model._set_of_time_periods, domain=pyo.Binary)
    model._valve_status = pyo.Var(model._set_of_valves, model._set_of_time_periods, domain=pyo.Binary)

def DefineObjective(model):
    model.OBJ = pyo.Objective(rule=_objective_function, sense=pyo.minimize)

def AddHeadLossConstraints(model, head_loss_equation, epsilon):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    model._cons_head_loss = pyo.Constraint(model._set_of_pipes, model._set_of_time_periods, rule=lambda model, pipe, t: model._head[model._pipes[pipe].start_node.label,t] - model._head[model._pipes[pipe].end_node.label,t] == model._pipes[pipe].loss_coefficient * model._flow[pipe,t] * (((model._flow[pipe,t] ** 2) + epsilon) ** ((exponent - 1) / 2)))

def AddPumpFlowConstraints(model):
    model._cons_pump_flow_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._flow[pump,t] >= model._pumps[pump].flow_lower_bound[t-1] * model._pump_status[pump,t])
    model._cons_pump_flow_u = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._pumps[pump].flow_upper_bound[t-1] * model._pump_status[pump,t] >= model._flow[pump,t])

def AddValveFlowConstraints(model):
    model._cons_valve_flow_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._flow[valve,t] >= model._valves[valve].flow_lower_bound[t-1] * model._valve_status[valve,t])
    model._cons_valve_flow_u = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._valve_status[valve,t] * model._valves[valve].flow_upper_bound[t-1] >= model._flow[valve,t])

def AddPumpHeadConstraints(model, epsilon):
    model._cons_pump_head_l = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + (model._pumps[pump].curve_b * ((model._flow[pump,t] + epsilon) ** model._pumps[pump].curve_c)) >= (model._pumps[pump].start_node.head_lower_bound[t-1] - model._pumps[pump].end_node.head_upper_bound[t-1]) * (1 - model._pump_status[pump,t]))
    model._cons_pump_head_u = pyo.Constraint(model._set_of_pumps, model._set_of_time_periods, rule=lambda model, pump, t: (model._pumps[pump].start_node.head_upper_bound[t-1] - model._pumps[pump].end_node.head_lower_bound[t-1]) * (1 - model._pump_status[pump,t]) >= model._head[model._pumps[pump].start_node.label,t] - model._head[model._pumps[pump].end_node.label,t] + (model._pumps[pump].curve_a * model._pump_status[pump,t]) + (model._pumps[pump].curve_b * ((model._flow[pump,t] + epsilon) ** model._pumps[pump].curve_c)))

def AddValveHeadConstraints(model):
    model._cons_valve_head_l = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t] >= (model._valves[valve].start_node.head_lower_bound[t-1] - model._valves[valve].end_node.head_upper_bound[t-1]) * (1 - model._valve_status[valve,t]))
    model._cons_valve_head_u = pyo.Constraint(model._set_of_valves, model._set_of_time_periods, rule=lambda model, valve, t: (model._valves[valve].start_node.head_upper_bound[t-1] - model._valves[valve].end_node.head_lower_bound[t-1]) * (1 - model._valve_status[valve,t]) >= model._head[model._valves[valve].start_node.label,t] - model._head[model._valves[valve].end_node.label,t])

def AddTankBalanceConstraints(model, hydraulic_timestep): #m^3 / s == m^3
    model._cons_tank_balance = pyo.Constraint(model._set_of_tanks, model._set_of_time_periods, rule=lambda model, tank, t: hydraulic_timestep * sum(model._flow[edge.label,t] for edge in model._tanks[tank].incoming_edges) - hydraulic_timestep * sum(model._flow[edge.label,t] for edge in model._tanks[tank].outgoing_edges) == model._storage[tank,t] - model._storage[tank,t-1])

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    model._cons_junction_balance = pyo.Constraint(model._set_of_non_reservoir_junctions, model._set_of_time_periods, rule=lambda model, junction, t: sum(model._flow[edge.label,t] for edge in model._junctions[junction].incoming_edges) - sum(model._flow[edge.label,t] for edge in model._junctions[junction].outgoing_edges) == model._junctions[junction].net_demand[t-1])

def AddTankHeadConstraints(model):
    model._cons_tank_head = pyo.Constraint(model._set_of_tanks, model._set_of_time_periods, rule=lambda model, tank, t: model._head[tank,t] == model._tanks[tank].elevation + ((1 / model._tanks[tank].area) * model._storage[tank,t]))

def PyomoOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    model = pyo.ConcreteModel()
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model, head_loss_equation, 1e-6)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
    AddPumpHeadConstraints(model, 1e-6)
    AddValveHeadConstraints(model)
    AddTankBalanceConstraints(model, hydraulic_timestep)
    AddJunctionBalanceConstraints(model)
    AddTankHeadConstraints(model)

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, f'{epanet_model}_formulation.txt'), 'w') as f:
        model.pprint(ostream=f)

    solver = pyo.SolverFactory('baron', executable='/Users/emirsener/baron-osxarm64/baron')
    solver.options['CplexLibName'] = '/Users/emirsener/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/libcplex2212.dylib'
    solver.options['MaxTime'] = 3600
    solver.options['LPSol'] = 3

    log_path = os.path.join(results_dir, f'{epanet_model}_baron_log.txt')
    with tempfile.TemporaryDirectory(prefix='pyomo_baron_') as scratch:
        TempfileManager.tempdir = scratch
        results = solver.solve(model, tee=True, logfile=log_path)

    with open(os.path.join(results_dir, f'{epanet_model}_solution.txt'), 'w') as f:
        model.display(ostream=f)