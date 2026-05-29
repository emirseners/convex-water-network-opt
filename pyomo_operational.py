import os
import tempfile
import pyomo.environ as pyo
from pyomo.common.tempfiles import TempfileManager

def _headbounds(model, i, t):
    return model.nodes[i].head_lower_bound[t-1], model.nodes[i].head_upper_bound[t-1]

def _flowbounds(model, i, t):
    return model.links[i].flow_lower_bound[t-1], model.links[i].flow_upper_bound[t-1]

def _storage_bounds(model, i, t):
    if t == 0:
        return model.tanks[i].storage_initial, model.tanks[i].storage_initial
    else:
        return model.tanks[i].storage_lower_bound, model.tanks[i].storage_upper_bound

def _objective_function(model):
    return pyo.quicksum((model.pumps[i].fixed_cost[t-1] * model.pump_status[i, t]) + (model.pumps[i].operational_cost[t-1] * model.flow[i, t]) for i in model.set_of_pumps for t in model.set_of_time_periods)

def DefineSets(model):
    model.set_of_junctions = pyo.Set(initialize=list(model.junctions.keys()))
    model.set_of_non_reservoir_junctions = pyo.Set(initialize=[label for label, junction in model.junctions.items() if junction.segment != 'reservoir'])
    model.set_of_tanks = pyo.Set(initialize=list(model.tanks.keys()))
    model.set_of_pipes = pyo.Set(initialize=list(model.pipes.keys()))
    model.set_of_pumps = pyo.Set(initialize=list(model.pumps.keys()))
    model.set_of_valves = pyo.Set(initialize=list(model.valves.keys()))
    model.set_of_time_periods = pyo.RangeSet(1, model.time_steps)
    model.set_of_time_periods_with_initial = pyo.RangeSet(0, model.time_steps)
    model.set_of_nodes = pyo.Set(initialize=list(model.nodes.keys()))
    model.set_of_links = pyo.Set(initialize=list(model.links.keys()))

def DefineDecisionVariables(model):
    model.head = pyo.Var(model.set_of_nodes, model.set_of_time_periods, bounds=_headbounds)
    model.flow = pyo.Var(model.set_of_links, model.set_of_time_periods, bounds=_flowbounds)
    model.storage = pyo.Var(model.set_of_tanks, model.set_of_time_periods_with_initial, bounds=_storage_bounds)
    model.pump_status = pyo.Var(model.set_of_pumps, model.set_of_time_periods, domain=pyo.Binary)
    model.valve_status = pyo.Var(model.set_of_valves, model.set_of_time_periods, domain=pyo.Binary)

def DefineObjective(model):
    model.OBJ = pyo.Objective(rule=_objective_function, sense=pyo.minimize)

def AddHeadLossConstraints(model, head_loss_equation, epsilon = 0.0):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    model.cons_head_loss = pyo.Constraint(model.set_of_pipes, model.set_of_time_periods, rule=lambda m, pipe, t: (m.head[m.pipes[pipe].start_node.label, t] - m.head[m.pipes[pipe].end_node.label, t] - (m.pipes[pipe].loss_coefficient * m.flow[pipe, t] * (((m.flow[pipe, t] ** 2) + epsilon) ** exponent)) == 0))

def AddPumpFlowConstraints(model):
    model.cons_pump_flow_l = pyo.Constraint(model.set_of_pumps, model.set_of_time_periods, rule=lambda m, pump, t: (m.flow[pump, t] - (m.pump_status[pump, t] * m.pumps[pump].flow_lower_bound[t-1]) >= 0))
    model.cons_pump_flow_u = pyo.Constraint(model.set_of_pumps, model.set_of_time_periods, rule=lambda m, pump, t: ((m.pump_status[pump, t] * m.pumps[pump].flow_upper_bound[t-1]) - m.flow[pump, t] >= 0))

def AddValveFlowConstraints(model):
    model.cons_valve_flow_l = pyo.Constraint(model.set_of_valves, model.set_of_time_periods, rule=lambda m, valve, t: (m.flow[valve, t] - (m.valve_status[valve, t] * m.valves[valve].flow_lower_bound[t-1]) >= 0))
    model.cons_valve_flow_u = pyo.Constraint(model.set_of_valves, model.set_of_time_periods, rule=lambda m, valve, t: ((m.valve_status[valve, t] * m.valves[valve].flow_upper_bound[t-1]) - m.flow[valve, t] >= 0))

def AddPumpHeadConstraints(model):
    pass

def AddValveHeadConstraints(model):
    model.cons_valve_head_l = pyo.Constraint(model.set_of_valves, model.set_of_time_periods, rule=lambda m, valve, t: (m.head[m.valves[valve].start_node.label, t] - m.head[m.valves[valve].end_node.label, t] - ((m.valves[valve].start_node.head_lower_bound[t-1] - m.valves[valve].end_node.head_upper_bound[t-1]) * (1 - m.valve_status[valve, t])) >= 0))
    model.cons_valve_head_u = pyo.Constraint(model.set_of_valves, model.set_of_time_periods, rule=lambda m, valve, t: (((m.valves[valve].start_node.head_upper_bound[t-1] - m.valves[valve].end_node.head_lower_bound[t-1]) * (1 - m.valve_status[valve, t])) - m.head[m.valves[valve].start_node.label, t] + m.head[m.valves[valve].end_node.label, t] >= 0))

def AddTankBalanceConstraints(model, hydraulic_timestep): #m^3 / s == m^3
    model.cons_tank_balance = pyo.Constraint(model.set_of_tanks, model.set_of_time_periods, rule=lambda m, tank, t: (hydraulic_timestep * sum(m.flow[edge.label, t] for edge in m.tanks[tank].incoming_edges) - hydraulic_timestep * sum(m.flow[edge.label, t] for edge in m.tanks[tank].outgoing_edges) - m.storage[tank, t] + m.storage[tank, t-1] == 0))

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    model.cons_junction_balance = pyo.Constraint(model.set_of_non_reservoir_junctions, model.set_of_time_periods, rule=lambda m, junction, t: (sum(m.flow[edge.label, t] for edge in m.junctions[junction].incoming_edges) - sum(m.flow[edge.label, t] for edge in m.junctions[junction].outgoing_edges) - m.junctions[junction].net_demand[t-1] == 0))

def AddTankHeadConstraints(model):
    model.cons_tank_head = pyo.Constraint(model.set_of_tanks,  model.set_of_time_periods, rule=lambda m, tank, t: (m.head[tank, t] == m.tanks[tank].elevation + ((1 / m.tanks[tank].area) * m.storage[tank, t])))

def PyomoOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    model = pyo.ConcreteModel()
    model.junctions, model.tanks, model.pipes, model.pumps, model.valves, model.time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model.nodes, model.links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model)
    DefineObjective(model)
    AddHeadLossConstraints(model, head_loss_equation, 1e-6)
    AddPumpFlowConstraints(model)
    AddValveFlowConstraints(model)
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