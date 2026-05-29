from mosek.fusion import *
import mosek.fusion.pythonic
import numpy as np
import sys

def DefineSets(model):
    model.set_of_junctions = list(model.junctions.keys())
    model.set_of_tanks = list(model.tanks.keys())
    model.set_of_pipes = list(model.pipes.keys())
    model.set_of_pumps = list(model.pumps.keys())
    model.set_of_valves = list(model.valves.keys())
    model.set_of_time_periods = list(range(0, model.time_steps))
    model.set_of_time_periods_with_initial = list(range(0, model.time_steps + 1))
    model.set_of_nodes = list(model.nodes.keys())
    model.set_of_links = list(model.links.keys())

def DefineDecisionVariables(model):
    
    model.head = model.variable('head', [len(model.set_of_nodes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.nodes[n].head_lower_bound for n in model.set_of_nodes]), np.stack([model.nodes[n].head_upper_bound for n in model.set_of_nodes])))

    model.flow = model.variable('flow', [len(model.set_of_links), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.links[l].flow_lower_bound for l in model.set_of_links]), np.stack([model.links[l].flow_upper_bound for l in model.set_of_links])))

    if len(model.set_of_pumps) > 0:
        model.pump_status = model.variable('pump_status', [len(model.set_of_pumps), len(model.set_of_time_periods)],
                                    Domain.binary())

    if len(model.set_of_valves) > 0:
        model.valve_status = model.variable('valve_status', [len(model.set_of_valves), len(model.set_of_time_periods)],
                                    Domain.binary())

    model.flow_pipe_plus = model.variable('flow_pipe_plus', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.zeros_like(np.stack([model.pipes[l].flow_lower_bound for l in model.set_of_pipes])), np.stack([model.pipes[l].flow_upper_bound for l in model.set_of_pipes])))

    model.flow_pipe_minus = model.variable('flow_pipe_minus', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.pipes[l].flow_lower_bound for l in model.set_of_pipes]), np.zeros_like(np.stack([model.pipes[l].flow_upper_bound for l in model.set_of_pipes]))))

    model.head_pipe_start_plus = model.variable('head_pipe_start_plus', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.pipes[p].start_node.head_lower_bound for p in model.set_of_pipes]), np.stack([model.pipes[p].start_node.head_upper_bound for p in model.set_of_pipes])))

    model.head_pipe_start_minus = model.variable('head_pipe_start_minus', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.pipes[p].start_node.head_lower_bound for p in model.set_of_pipes]), np.stack([model.pipes[p].start_node.head_upper_bound for p in model.set_of_pipes])))

    model.head_pipe_end_plus = model.variable('head_pipe_end_plus', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.pipes[p].end_node.head_lower_bound for p in model.set_of_pipes]), np.stack([model.pipes[p].end_node.head_upper_bound for p in model.set_of_pipes])))

    model.head_pipe_end_minus = model.variable('head_pipe_end_minus', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.inRange(np.stack([model.pipes[p].end_node.head_lower_bound for p in model.set_of_pipes]), np.stack([model.pipes[p].end_node.head_upper_bound for p in model.set_of_pipes])))

    model.flow_pipe_plus_status = model.variable('flow_pipe_plus_status', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.binary())

    model.flow_pipe_minus_status = model.variable('flow_pipe_minus_status', [len(model.set_of_pipes), len(model.set_of_time_periods)],
                                    Domain.binary())

    if len(model.set_of_tanks) > 0:
        model.storage = model.variable('storage', [len(model.set_of_tanks), len(model.set_of_time_periods_with_initial)],
                                    Domain.inRange(np.stack([np.concatenate([[model.tanks[k].storage_initial], np.full(model.time_steps, model.tanks[k].storage_lower_bound)]) for k in model.set_of_tanks]), np.stack([np.concatenate([[model.tanks[k].storage_initial], np.full(model.time_steps, model.tanks[k].storage_upper_bound)]) for k in model.set_of_tanks])))

def DefineObjective(model):
    model.objective('objective', ObjectiveSense.Minimize, sum(pump.operational_cost[t] * model.flow[model.set_of_links.index(pump.label), t] + pump.fixed_cost[t] * model.pump_status[i, t] for i, pump in enumerate(model.pumps.values()) for t in model.set_of_time_periods))

def AddHeadLossConstraints(model, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2

    for i, pipe in enumerate(model.pipes.values()):
        for t in model.set_of_time_periods:
            model.constraint(f'cons_flow_plus_minus_{pipe.label}_{t}', model.flow[model.set_of_links.index(pipe.label), t] == model.flow_pipe_plus[i, t] + model.flow_pipe_minus[i, t])

    for i, pipe in enumerate(model.pipes.values()):
        for t in model.set_of_time_periods:
            model.constraint(f'cons_head_start_plus_minus_{pipe.label}_{t}', model.head[model.set_of_nodes.index(pipe.start_node.label), t] == model.head_pipe_start_plus[i, t] + model.head_pipe_start_minus[i, t])
            model.constraint(f'cons_head_end_plus_minus_{pipe.label}_{t}', model.head[model.set_of_nodes.index(pipe.end_node.label), t] == model.head_pipe_end_plus[i, t] + model.head_pipe_end_minus[i, t])

    for i, pipe in enumerate(model.pipes.values()):
        for t in model.set_of_time_periods:
            model.constraint(f'cons_flow_pipe_status_{pipe.label}_{t}', model.flow_pipe_minus_status[i, t] + model.flow_pipe_plus_status[i, t] == 1)

    for i, pipe in enumerate(model.pipes.values()):
        beta = pipe.loss_coefficient ** (1/(1-exponent))
        for t in model.set_of_time_periods:    
            model.constraint(Expr.vstack(model.head_pipe_start_plus[i, t] - model.head_pipe_end_plus[i, t], beta * model.flow_pipe_plus_status[i, t], model.flow_pipe_plus[i, t]),
                Domain.inPPowerCone(1/exponent))
            model.constraint(Expr.vstack(model.head_pipe_start_minus[i, t] - model.head_pipe_end_minus[i, t], beta * model.flow_pipe_minus_status[i, t], model.flow_pipe_minus[i, t]),
                Domain.inPPowerCone(1/exponent))

def AddPumpFlowConstraints(model):
    for i, pump in enumerate(model.pumps.values()):
        for t in model.set_of_time_periods:
            model.constraint(f'cons_pump_flow_u_{pump.label}_{t}', (model.pump_status[i, t] * pump.flow_upper_bound[t]) - model.flow[model.set_of_links.index(pump.label), t] >= 0)
            model.constraint(f'cons_pump_flow_l_{pump.label}_{t}', model.flow[model.set_of_links.index(pump.label), t] - (model.pump_status[i, t] * pump.flow_lower_bound[t]) >= 0)

def AddValveFlowConstraints(model):
    for i, valve in enumerate(model.valves.values()):
        for t in model.set_of_time_periods:
            model.constraint(f'cons_valve_flow_u_{valve.label}_{t}', (model.valve_status[i, t] * valve.flow_upper_bound[t]) - model.flow[model.set_of_links.index(valve.label), t] >= 0)
            model.constraint(f'cons_valve_flow_l_{valve.label}_{t}', model.flow[model.set_of_links.index(valve.label), t] - (model.valve_status[i, t] * valve.flow_lower_bound[t]) >= 0)

def AddPumpHeadConstraints(model):
    pass

def AddValveHeadConstraints(model):
    for i, valve in enumerate(model.valves.values()):
        for t in model.set_of_time_periods:
            model.constraint(f'cons_valve_head_u_{valve.label}_{t}', ((1 - model.valve_status[i, t]) * (valve.start_node.head_upper_bound[t] - valve.end_node.head_lower_bound[t])) - model.head[model.set_of_nodes.index(valve.start_node.label), t] + model.head[model.set_of_nodes.index(valve.end_node.label), t] >= 0)
            model.constraint(f'cons_valve_head_l_{valve.label}_{t}', model.head[model.set_of_nodes.index(valve.start_node.label), t] - model.head[model.set_of_nodes.index(valve.end_node.label), t] - ((1 - model.valve_status[i, t]) * (valve.start_node.head_lower_bound[t] - valve.end_node.head_upper_bound[t])) >= 0)

def AddTankBalanceConstraints(model, hydraulic_timestep): #m^3 / s == m^3
    for i, tank in enumerate(model.tanks.values()):    
        for t in model.set_of_time_periods:
            model.constraint(f'cons_tank_balance_{tank.label}_{t}', hydraulic_timestep * sum(model.flow[model.set_of_links.index(edge.label), t] for edge in tank.incoming_edges) - hydraulic_timestep * sum(model.flow[model.set_of_links.index(edge.label), t] for edge in tank.outgoing_edges) - model.storage[i, t+1] + model.storage[i, t] == 0)

def AddJunctionBalanceConstraints(model): #m^3 / s == m^3
    for junction in model.junctions.values():
        for t in model.set_of_time_periods:
            if junction.segment == 'reservoir':
                continue
            model.constraint(f'cons_junction_balance_{junction.label}_{t}', sum(model.flow[model.set_of_links.index(edge.label), t] for edge in junction.incoming_edges) - sum(model.flow[model.set_of_links.index(edge.label), t] for edge in junction.outgoing_edges) - junction.net_demand[t] == 0)

def AddTankHeadConstraints(model):
    for i, tank in enumerate(model.tanks.values()):    
        for t in model.set_of_time_periods:
            model.constraint(f'cons_tank_head_{tank.label}_{t+1}', model.head[model.set_of_nodes.index(tank.label), t] == tank.elevation + ((1 / tank.area) * model.storage[i, t+1]))

def MosekOperationalModel(epanet_model, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_loss_equation):
    model = Model(epanet_model)
    model.junctions, model.tanks, model.pipes, model.pumps, model.valves, model.time_steps = junctions, tanks, pipes, pumps, valves, time_steps
    model.nodes, model.links = junctions | tanks, pipes | pumps | valves

    DefineSets(model)
    DefineDecisionVariables(model)
    AddHeadLossConstraints(model, head_loss_equation)
    AddJunctionBalanceConstraints(model)

    if len(model.set_of_pumps) > 0:
        DefineObjective(model)
        AddPumpFlowConstraints(model)
    
    if len(model.set_of_valves) > 0:
        AddValveFlowConstraints(model)
        AddValveHeadConstraints(model)

    if len(model.set_of_tanks) > 0:
        AddTankBalanceConstraints(model, hydraulic_timestep)
        AddTankHeadConstraints(model)

    model.writeTask("model.ptf")
    model.setLogHandler(sys.stdout)
    model.solve()