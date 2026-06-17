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
    if len(model._set_of_nodes) > 0:
        model._head = model.variable('head', [len(model._set_of_nodes), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([model._nodes[n].head_lower_bound for n in model._set_of_nodes]), np.stack([model._nodes[n].head_upper_bound for n in model._set_of_nodes])))

    if len(model._set_of_links) > 0:
        model._flow = model.variable('flow', [len(model._set_of_links), len(model._set_of_time_periods)], Domain.inRange(
            np.stack([model._links[l].flow_lower_bound for l in model._set_of_links]), np.stack([model._links[l].flow_upper_bound for l in model._set_of_links])))

    if len(model._set_of_tanks) > 0:
        model._storage = model.variable('storage', [len(model._set_of_tanks), len(model._set_of_time_periods_with_initial)], Domain.inRange(
            np.stack([np.concatenate([[model._tanks[k].storage_initial], np.full(model._time_steps, model._tanks[k].storage_lower_bound)]) for k in model._set_of_tanks]), np.stack([np.concatenate([[model._tanks[k].storage_initial], np.full(model._time_steps, model._tanks[k].storage_upper_bound)]) for k in model._set_of_tanks])))

    if len(model._set_of_pumps) > 0:
        model._pump_status = model.variable('pump_status', [len(model._set_of_pumps), len(model._set_of_time_periods)], Domain.binary())

    if len(model._set_of_valves) > 0:
        model._valve_status = model.variable('valve_status', [len(model._set_of_valves), len(model._set_of_time_periods)], Domain.binary())

    if len(model._set_of_pipes) > 0:
        model._flow_pipe_plus = model.variable('flow_pipe_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
            Domain.inRange(np.zeros((len(model._set_of_pipes), len(model._set_of_time_periods))), np.stack([model._pipes[p].flow_upper_bound for p in model._set_of_pipes])))

        model._flow_pipe_minus = model.variable('flow_pipe_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
            Domain.inRange(np.stack([model._pipes[p].flow_lower_bound for p in model._set_of_pipes]), np.zeros((len(model._set_of_pipes), len(model._set_of_time_periods)))))

        model._head_pipe_diff_plus = model.variable('head_pipe_diff_plus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
            Domain.inRange(np.zeros((len(model._set_of_pipes), len(model._set_of_time_periods))), np.stack([np.maximum(model._pipes[p].start_node.head_upper_bound - model._pipes[p].end_node.head_lower_bound, 0) for p in model._set_of_pipes])))

        model._head_pipe_diff_minus = model.variable('head_pipe_diff_minus', [len(model._set_of_pipes), len(model._set_of_time_periods)],
            Domain.inRange(np.stack([np.minimum(model._pipes[p].start_node.head_lower_bound - model._pipes[p].end_node.head_upper_bound, 0) for p in model._set_of_pipes]), np.zeros((len(model._set_of_pipes), len(model._set_of_time_periods)))))

        model._flow_pipe_status = model.variable('flow_pipe_status', [len(model._set_of_pipes), len(model._set_of_time_periods)], Domain.binary())

    if len(model._set_of_pumps) > 0:
        model._head_gain_flow = model.variable('head_gain_flow', [len(model._set_of_pumps), len(model._set_of_time_periods)],
            Domain.inRange(np.zeros((len(model._set_of_pumps), len(model._set_of_time_periods))), np.stack([model._pumps[p].flow_upper_bound ** model._pumps[p].curve_c for p in model._set_of_pumps])))

def DefineObjective(model):
    if len(model._set_of_pumps) > 0:
        model.objective('objective', ObjectiveSense.Minimize, sum(pump.operational_cost[t] * model._flow[model._set_of_links.index(pump.label), t] + pump.fixed_cost[t] * model._pump_status[i, t] for i, pump in enumerate(model._pumps.values()) for t in model._set_of_time_periods))
    else:
        model.objective('objective', ObjectiveSense.Minimize, Expr.constTerm(0.0))

def AddHeadLossConstraints(model, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2

    for i, pipe in enumerate(model._pipes.values()):
        beta = pipe.loss_coefficient ** (1/(1-exponent))
        for t in model._set_of_time_periods:
            model.constraint(f'cons_flow_{pipe.label}_{t}', model._flow[model._set_of_links.index(pipe.label), t] == model._flow_pipe_plus[i, t] + model._flow_pipe_minus[i, t])
            model.constraint(f'cons_head_diff_{pipe.label}_{t}', model._head[model._set_of_nodes.index(pipe.start_node.label), t] - model._head[model._set_of_nodes.index(pipe.end_node.label), t] == model._head_pipe_diff_plus[i, t] + model._head_pipe_diff_minus[i, t])

            if pipe.flow_upper_bound[t] > 0:
                model.constraint(f'cons_plus_diff_{pipe.label}_{t}', model._head_pipe_diff_plus[i, t]  <= model._flow_pipe_status[i, t] * max(0.0, pipe.start_node.head_upper_bound[t] - pipe.end_node.head_lower_bound[t]))
                model.constraint(f'cons_plus_flow_l_{pipe.label}_{t}', model._flow_pipe_plus[i, t] >= model._flow_pipe_status[i, t] * max(0.0, pipe.flow_lower_bound[t]))
                model.constraint(f'cons_plus_flow_u_{pipe.label}_{t}', model._flow_pipe_plus[i, t] <= model._flow_pipe_status[i, t] * pipe.flow_upper_bound[t])
                model.constraint(Expr.vstack(model._head_pipe_diff_plus[i, t], beta * model._flow_pipe_status[i, t], model._flow_pipe_plus[i, t]), Domain.inPPowerCone(1/exponent))

                model.constraint(f'cons_secant_plane_plus_{pipe.label}_{t}', model._head_pipe_diff_plus[i, t] <= (pipe.loss_coefficient * (((pipe.flow_upper_bound[t] ** exponent) - (max(0.0, pipe.flow_lower_bound[t]) ** exponent)) / (pipe.flow_upper_bound[t] - max(0.0, pipe.flow_lower_bound[t]))) * (model._flow_pipe_plus[i, t] - (max(0.0, pipe.flow_lower_bound[t]) * model._flow_pipe_status[i, t]))) + (pipe.loss_coefficient * (max(0.0, pipe.flow_lower_bound[t]) ** exponent) * model._flow_pipe_status[i, t]))

            else:
                model.constraint(f"fix_plus_diff_{pipe.label}_{t}", model._head_pipe_diff_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f"fix_plus_flow_{pipe.label}_{t}", model._flow_pipe_plus[i, t], Domain.equalsTo(0.0))
                model.constraint(f"fix_status_{pipe.label}_{t}", model._flow_pipe_status[i, t], Domain.equalsTo(0))

            if pipe.flow_lower_bound[t] < 0:
                model.constraint(f'cons_minus_diff_{pipe.label}_{t}', model._head_pipe_diff_minus[i, t]  >= (1 - model._flow_pipe_status[i, t]) * min(0.0, pipe.start_node.head_lower_bound[t] - pipe.end_node.head_upper_bound[t]))
                model.constraint(f'cons_minus_flow_l_{pipe.label}_{t}', model._flow_pipe_minus[i, t] >= (1 - model._flow_pipe_status[i, t]) * pipe.flow_lower_bound[t])
                model.constraint(f'cons_minus_flow_u_{pipe.label}_{t}', model._flow_pipe_minus[i, t] <= (1 - model._flow_pipe_status[i, t]) * min(0.0, pipe.flow_upper_bound[t]))
                model.constraint(Expr.vstack(-model._head_pipe_diff_minus[i, t], beta * (1 - model._flow_pipe_status[i, t]), model._flow_pipe_minus[i, t]), Domain.inPPowerCone(1/exponent))

                model.constraint(f'cons_secant_plane_minus_{pipe.label}_{t}', model._head_pipe_diff_minus[i, t] >= (pipe.loss_coefficient * ((((-pipe.flow_lower_bound[t]) ** exponent) - ((-min(0.0, pipe.flow_upper_bound[t])) ** exponent)) / (min(0.0, pipe.flow_upper_bound[t]) - pipe.flow_lower_bound[t])) * (model._flow_pipe_minus[i, t] - (pipe.flow_lower_bound[t] * (1 - model._flow_pipe_status[i, t])))) - (pipe.loss_coefficient * ((-pipe.flow_lower_bound[t]) ** exponent) * (1 - model._flow_pipe_status[i, t])))

            else:
                model.constraint(f"fix_minus_diff_{pipe.label}_{t}", model._head_pipe_diff_minus[i, t], Domain.equalsTo(0.0))
                model.constraint(f"fix_minus_flow_{pipe.label}_{t}", model._flow_pipe_minus[i, t], Domain.equalsTo(0.0))
                if pipe.flow_upper_bound[t] > 0:
                    model.constraint(f"fix_status_{pipe.label}_{t}", model._flow_pipe_status[i, t], Domain.equalsTo(1))

def AddValveFlowConstraints(model):
    for i, valve in enumerate(model._valves.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_valve_flow_u_{valve.label}_{t}', (model._valve_status[i, t] * valve.flow_upper_bound[t]) - model._flow[model._set_of_links.index(valve.label), t] >= 0)
            model.constraint(f'cons_valve_flow_l_{valve.label}_{t}', model._flow[model._set_of_links.index(valve.label), t] - (model._valve_status[i, t] * valve.flow_lower_bound[t]) >= 0)

def AddPumpHeadConstraints(model):
    for i, pump in enumerate(model._pumps.values()):
        for t in model._set_of_time_periods:
            model.constraint(f'cons_head_gain_flow_l_{pump.label}_{t}', model._head_gain_flow[i, t] >= (pump.flow_lower_bound[t] ** pump.curve_c) * model._pump_status[i, t])
            model.constraint(f'cons_head_gain_flow_u_{pump.label}_{t}', model._head_gain_flow[i, t] <= (pump.flow_upper_bound[t] ** pump.curve_c) * model._pump_status[i, t])
            model.constraint(f'cons_head_gain_l_{pump.label}_{t}', model._head[model._set_of_nodes.index(pump.start_node.label),t] - model._head[model._set_of_nodes.index(pump.end_node.label),t] + (pump.curve_a * model._pump_status[i, t]) + (pump.curve_b * model._head_gain_flow[i, t]) >= (pump.start_node.head_lower_bound[t] - pump.end_node.head_upper_bound[t]) * (1 - model._pump_status[i, t]))
            model.constraint(f'cons_head_gain_u_{pump.label}_{t}', model._head[model._set_of_nodes.index(pump.start_node.label),t] - model._head[model._set_of_nodes.index(pump.end_node.label),t] + (pump.curve_a * model._pump_status[i, t]) + (pump.curve_b * model._head_gain_flow[i, t]) <= (pump.start_node.head_upper_bound[t] - pump.end_node.head_lower_bound[t]) * (1 - model._pump_status[i, t]))
            model.constraint(Expr.vstack(model._head_gain_flow[i, t], model._pump_status[i, t], model._flow[model._set_of_links.index(pump.label), t]), Domain.inPPowerCone(1/pump.curve_c))

            if pump.flow_lower_bound[t] != pump.flow_upper_bound[t]:
                model.constraint(f'cons_secant_plane_{pump.label}_{t}', model._head_gain_flow[i, t] <= model._pump_status[i, t] * (pump.flow_lower_bound[t] ** pump.curve_c) + (((pump.flow_upper_bound[t] ** pump.curve_c) - (pump.flow_lower_bound[t] ** pump.curve_c)) / (pump.flow_upper_bound[t] - pump.flow_lower_bound[t])) * (model._flow[model._set_of_links.index(pump.label), t] - (pump.flow_lower_bound[t] * model._pump_status[i, t])))

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
    DefineObjective(model)
    AddHeadLossConstraints(model, head_loss_equation)
    AddJunctionBalanceConstraints(model)

    if len(model._set_of_pumps) > 0:
        AddPumpHeadConstraints(model)
    
    if len(model._set_of_valves) > 0:
        AddValveFlowConstraints(model)
        AddValveHeadConstraints(model)

    if len(model._set_of_tanks) > 0:
        AddTankBalanceConstraints(model, hydraulic_timestep)
        AddTankHeadConstraints(model)

    #model.writeTask("results/mosek_model.ptf")
    model.setLogHandler(sys.stdout)
    model.setSolverParam("mioMaxTime", 3600)

    with open("results/mosek_solver.log", "w") as f:
        model.setLogHandler(f)
        model.solve()