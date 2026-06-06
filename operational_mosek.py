from mosek.fusion import *
import mosek.fusion.pythonic
import numpy as np
import sys

def DefineSets(model):
    model._set_of_junctions = list(model._junctions.keys())
    model._set_of_tanks = list(model._tanks.keys())
    model._set_of_pipes = list(model._pipes.keys())
    model._set_of_pumps = list(model._pumps.keys())
    model._set_of_valves = list(model._valves.keys())
    model._set_of_time_periods = list(range(0, model._time_steps))
    model._set_of_time_periods_with_initial = list(range(0, model._time_steps + 1))
    model._set_of_nodes = list(model._nodes.keys())
    model._set_of_links = list(model._links.keys())

def DefineDecisionVariables(model):
    
    model._head = model.variable('head', [len(model._set_of_nodes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._nodes[n].head_lower_bound for n in model._set_of_nodes]), np.stack([model._nodes[n].head_upper_bound for n in model._set_of_nodes])))

    model._flow = model.variable('flow', [len(model._set_of_links), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._links[l].flow_lower_bound for l in model._set_of_links]), np.stack([model._links[l].flow_upper_bound for l in model._set_of_links])))

    if len(model._set_of_pumps) > 0:
        model._pump_status = model.variable('pump_status', [len(model._set_of_pumps), len(model._set_of_time_periods)],
                                    Domain.binary())

    if len(model._set_of_valves) > 0:
        model._valve_status = model.variable('valve_status', [len(model._set_of_valves), len(model._set_of_time_periods)],
                                    Domain.binary())

    model._flow_pipe_plus = model.variable('flow_pipe_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.zeros_like(np.stack([model._pipes[l].flow_lower_bound for l in model._set_of_pipes])), np.stack([model._pipes[l].flow_upper_bound for l in model._set_of_pipes])))

    model._flow_pipe_minus = model.variable('flow_pipe_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._pipes[l].flow_lower_bound for l in model._set_of_pipes]), np.zeros_like(np.stack([model._pipes[l].flow_upper_bound for l in model._set_of_pipes]))))

    model._head_pipe_start_plus = model.variable('head_pipe_start_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._pipes[p].start_node.head_lower_bound for p in model._set_of_pipes]), np.stack([model._pipes[p].start_node.head_upper_bound for p in model._set_of_pipes])))

    model._head_pipe_start_minus = model.variable('head_pipe_start_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._pipes[p].start_node.head_lower_bound for p in model._set_of_pipes]), np.stack([model._pipes[p].start_node.head_upper_bound for p in model._set_of_pipes])))

    model._head_pipe_end_plus = model.variable('head_pipe_end_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._pipes[p].end_node.head_lower_bound for p in model._set_of_pipes]), np.stack([model._pipes[p].end_node.head_upper_bound for p in model._set_of_pipes])))

    model._head_pipe_end_minus = model.variable('head_pipe_end_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.inRange(np.stack([model._pipes[p].end_node.head_lower_bound for p in model._set_of_pipes]), np.stack([model._pipes[p].end_node.head_upper_bound for p in model._set_of_pipes])))

    model._flow_pipe_plus_status = model.variable('flow_pipe_plus_status', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.binary())

    model._flow_pipe_minus_status = model.variable('flow_pipe_minus_status', [len(model._set_of_pipes), len(model._set_of_time_periods)],
                                    Domain.binary())

    if len(model._set_of_tanks) > 0:
        model._storage = model.variable('storage', [len(model._set_of_tanks), len(model._set_of_time_periods_with_initial)],
                                    Domain.inRange(np.stack([np.concatenate([[model._tanks[k].storage_initial], np.full(model._time_steps, model._tanks[k].storage_lower_bound)]) for k in model._set_of_tanks]), np.stack([np.concatenate([[model._tanks[k].storage_initial], np.full(model._time_steps, model._tanks[k].storage_upper_bound)]) for k in model._set_of_tanks])))

def DefineObjective(model):
    model.objective('objective', ObjectiveSense.Minimize, sum(pump.operational_cost[t] * model._flow[model._set_of_links.index(pump.label), t] + pump.fixed_cost[t] * model._pump_status[i, t] for i, pump in enumerate(model._pumps.values()) for t in model._set_of_time_periods))

def AddHeadLossConstraints(model, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2

    for i, pipe in enumerate(model._pipes.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_flow_plus_minus_{pipe.label}_{t}', model._flow[model._set_of_links.index(pipe.label), t] == model._flow_pipe_plus[i, t] + model._flow_pipe_minus[i, t])

    for i, pipe in enumerate(model._pipes.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_head_start_plus_minus_{pipe.label}_{t}', model._head[model._set_of_nodes.index(pipe.start_node.label), t] == model._head_pipe_start_plus[i, t] + model._head_pipe_start_minus[i, t])
            model.constraint(f'cons_head_end_plus_minus_{pipe.label}_{t}', model._head[model._set_of_nodes.index(pipe.end_node.label), t] == model._head_pipe_end_plus[i, t] + model._head_pipe_end_minus[i, t])

    for i, pipe in enumerate(model._pipes.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_flow_pipe_status_{pipe.label}_{t}', model._flow_pipe_minus_status[i, t] + model._flow_pipe_plus_status[i, t] == 1)

    for i, pipe in enumerate(model._pipes.values()):
        beta = pipe.loss_coefficient ** (1/(1-exponent))
        for t in model._set_of_time_periods:    
            model.constraint(Expr.vstack(model._head_pipe_start_plus[i, t] - model._head_pipe_end_plus[i, t], beta * model._flow_pipe_plus_status[i, t], model._flow_pipe_plus[i, t]),
                Domain.inPPowerCone(1/exponent))
            model.constraint(Expr.vstack(model._head_pipe_start_minus[i, t] - model._head_pipe_end_minus[i, t], beta * model._flow_pipe_minus_status[i, t], model._flow_pipe_minus[i, t]),
                Domain.inPPowerCone(1/exponent))

def AddPumpFlowConstraints(model):
    for i, pump in enumerate(model._pumps.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_pump_flow_u_{pump.label}_{t}', (model._pump_status[i, t] * pump.flow_upper_bound[t]) - model._flow[model._set_of_links.index(pump.label), t] >= 0)
            model.constraint(f'cons_pump_flow_l_{pump.label}_{t}', model._flow[model._set_of_links.index(pump.label), t] - (model._pump_status[i, t] * pump.flow_lower_bound[t]) >= 0)

def AddValveFlowConstraints(model):
    for i, valve in enumerate(model._valves.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_valve_flow_u_{valve.label}_{t}', (model._valve_status[i, t] * valve.flow_upper_bound[t]) - model._flow[model._set_of_links.index(valve.label), t] >= 0)
            model.constraint(f'cons_valve_flow_l_{valve.label}_{t}', model._flow[model._set_of_links.index(valve.label), t] - (model._valve_status[i, t] * valve.flow_lower_bound[t]) >= 0)

def AddPumpHeadConstraints(model):
    pass

def AddValveHeadConstraints(model):
    for i, valve in enumerate(model._valves.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_valve_head_u_{valve.label}_{t}', ((1 - model._valve_status[i, t]) * (valve.start_node.head_upper_bound[t] - valve.end_node.head_lower_bound[t])) - model._head[model._set_of_nodes.index(valve.start_node.label), t] + model._head[model._set_of_nodes.index(valve.end_node.label), t] >= 0)
            model.constraint(f'cons_valve_head_l_{valve.label}_{t}', model._head[model._set_of_nodes.index(valve.start_node.label), t] - model._head[model._set_of_nodes.index(valve.end_node.label), t] - ((1 - model._valve_status[i, t]) * (valve.start_node.head_lower_bound[t] - valve.end_node.head_upper_bound[t])) >= 0)

def AddTankBalanceConstraints(model, hydraulic_timestep): #m^3 / s == m^3
    for i, tank in enumerate(model._tanks.values()):    
        for t in model._set_of_time_periods:
            model.constraint(f'cons_tank_balance_{tank.label}_{t}', hydraulic_timestep * sum(model._flow[model._set_of_links.index(edge.label), t] for edge in tank.incoming_edges) - hydraulic_timestep * sum(model._flow[model._set_of_links.index(edge.label), t] for edge in tank.outgoing_edges) - model._storage[i, t+1] + model._storage[i, t] == 0)

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    for junction in model._junctions.values():
        for t in model._set_of_time_periods:
            if junction.segment == 'reservoir':
                continue
            model.constraint(f'cons_junction_balance_{junction.label}_{t}', sum(model._flow[model._set_of_links.index(edge.label), t] for edge in junction.incoming_edges) - sum(model._flow[model._set_of_links.index(edge.label), t] for edge in junction.outgoing_edges) - junction.net_demand[t] == 0)

def AddTankHeadConstraints(model):
    for i, tank in enumerate(model._tanks.values()):    
        for t in model._set_of_time_periods:
            model.constraint(f'cons_tank_head_{tank.label}_{t+1}', model._head[model._set_of_nodes.index(tank.label), t] == tank.elevation + ((1 / tank.area) * model._storage[i, t+1]))

def MosekOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    model = Model(epanet_model)
    model._junctions, model._tanks, model._pipes, model._pumps, model._valves, model._time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model._nodes, model._links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model)
    AddHeadLossConstraints(model, head_loss_equation)
    AddJunctionBalanceConstraints(model)

    if len(model._set_of_pumps) > 0:
        DefineObjective(model)
        AddPumpFlowConstraints(model)
    
    if len(model._set_of_valves) > 0:
        AddValveFlowConstraints(model)
        AddValveHeadConstraints(model)

    if len(model._set_of_tanks) > 0:
        AddTankBalanceConstraints(model, hydraulic_timestep)
        AddTankHeadConstraints(model)

    model.writeTask("model.ptf")
    model.setLogHandler(sys.stdout)
    model.solve()