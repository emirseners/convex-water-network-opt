from collections import deque
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

def _norm(values):
    return float(np.max(np.abs(values), initial=0.0))

def _gradient_solve(incidence, demand, boundary_term, loss_coefficient, exponent, offset, initial_flow, initial_head, min_gradient, flow_tol=1e-8):
    flow, head = initial_flow.copy(), initial_head.copy()
    flow_change = np.inf

    while flow_change > flow_tol:
        loss = offset + loss_coefficient * np.sign(flow) * np.abs(flow) ** exponent
        mass = incidence @ flow - demand
        energy = loss + incidence.T @ head + boundary_term
        gradient = np.maximum(loss_coefficient * exponent * np.maximum(np.abs(flow), 1e-12) ** (exponent - 1.0), min_gradient)
        inverse_gradient = 1.0 / gradient
        if incidence.shape[0]:
            matrix = (incidence.multiply(inverse_gradient) @ incidence.T).tocsc()
            rhs = mass - incidence @ (inverse_gradient * energy)
            head_change = splu(matrix).solve(rhs)
        else:
            head_change = np.empty(0)

        next_flow = flow - inverse_gradient * (energy + incidence.T @ head_change)
        flow_change = float(np.sum(np.abs(next_flow - flow))) / max(1.0, float(np.sum(np.abs(next_flow))))
        flow, head = next_flow, head + head_change

    return flow, head

def _prune_zero_demand_leaves(starts, ends, demand, fixed_nodes):
    incident = [set() for _ in demand]
    for edge, (start, end) in enumerate(zip(starts, ends)):
        incident[start].add(edge)
        incident[end].add(edge)

    active_nodes = np.ones(len(demand), dtype=bool)
    active_links = np.ones(len(starts), dtype=bool)
    leaves = deque(i for i in range(len(demand)) if i not in fixed_nodes and demand[i] == 0.0 and len(incident[i]) == 1)
    removed = []
    while leaves:
        leaf = leaves.popleft()
        if not active_nodes[leaf] or len(incident[leaf]) != 1: continue
        edge = next(iter(incident[leaf]))
        neighbor = ends[edge] if starts[edge] == leaf else starts[edge]
        if neighbor == leaf: continue
        active_nodes[leaf], active_links[edge] = False, False
        incident[leaf].remove(edge)
        incident[neighbor].remove(edge)
        removed.append((leaf, neighbor, edge, 1.0 if starts[edge] == leaf else -1.0))
        if neighbor not in fixed_nodes and demand[neighbor] == 0.0 and len(incident[neighbor]) == 1:
            leaves.append(neighbor)

    return active_nodes, np.flatnonzero(active_links), removed

def _solve_period(nodes, junctions, period_links, known_heads, solution, t, mass_tol, head_tol, min_gradient, flow_tol=1e-8):
    node_labels, link_labels = list(nodes), list(period_links)
    node_index = {label: i for i, label in enumerate(node_labels)}
    starts = np.array([node_index[obj.start_node.label] for obj, _ in period_links.values()], dtype=int)
    ends = np.array([node_index[obj.end_node.label] for obj, _ in period_links.values()], dtype=int)
    columns = np.arange(len(period_links))
    incidence = coo_matrix((np.r_[-np.ones(len(period_links)), np.ones(len(period_links))], (np.r_[starts, ends], np.r_[columns, columns])), shape=(len(nodes), len(period_links))).tocsr()

    junction_indices, demand = [], np.zeros(len(nodes))
    for label, junction in junctions.items():
        if junction.segment == 'reservoir': continue
        demand[node_index[label]] = junction.net_demand[t]
        junction_indices.append(node_index[label])

    active_nodes, active_links, removed = _prune_zero_demand_leaves(starts, ends, demand, {node_index[label] for label in known_heads})
    graph = coo_matrix((np.ones(2 * len(period_links)), (np.r_[starts, ends], np.r_[ends, starts])), shape=(len(nodes), len(nodes))).tocsr()
    component_count, components = connected_components(graph, directed=False)
    fixed_heads, components_without_fixed_head = known_heads.copy(), []
    for component in range(component_count):
        members = np.flatnonzero(components == component)
        if any(node_labels[i] in fixed_heads for i in members): continue
        reference = node_labels[members[active_nodes[members]][0]]
        fixed_heads[reference] = nodes[reference].head_upper_bound[t]
        components_without_fixed_head.append(members)

    unknown = np.array([i for i, label in enumerate(node_labels) if active_nodes[i] and label not in fixed_heads], dtype=int)
    heads = np.zeros(len(nodes))
    for label, head in fixed_heads.items():
        heads[node_index[label]] = head
    matrix, boundary_term = incidence[unknown, :][:, active_links], (incidence.T @ heads)[active_links]

    loss_coefficient = np.array([-obj.curve_b if component == 'pump' else obj.resistance_coefficient for obj, component in period_links.values()], dtype=float)
    exponent = np.array([obj.curve_c if component == 'pump' else obj.exponent for obj, component in period_links.values()], dtype=float)
    offset = np.array([-obj.curve_a if component == 'pump' else 0.0 for obj, component in period_links.values()], dtype=float)
    initial_flow = np.array([solution['flow'][link_labels[i]][t] for i in active_links], dtype=float)
    initial_head = np.array([solution['head'][node_labels[i]][t] for i in unknown], dtype=float)

    flow = np.zeros(len(period_links))
    flow[active_links], heads[unknown] = _gradient_solve(matrix, demand[unknown], boundary_term, loss_coefficient[active_links], exponent[active_links], offset[active_links], initial_flow, initial_head, min_gradient, flow_tol)
    for leaf, neighbor, edge, orientation in reversed(removed):
        heads[leaf] = heads[neighbor] + orientation * offset[edge]
    for members in components_without_fixed_head:
        lower = max(nodes[node_labels[i]].head_lower_bound[t] - heads[i] for i in members)
        upper = min(nodes[node_labels[i]].head_upper_bound[t] - heads[i] for i in members)
        heads[members] += min(max(0.0, lower), upper)

    mass = (incidence @ flow - demand)[junction_indices]
    energy = offset + loss_coefficient * np.sign(flow) * np.abs(flow) ** exponent + incidence.T @ heads
    if not (_norm(mass) <= mass_tol and _norm(energy) <= head_tol): return 'infeasible', None

    return 'feasible', {'head': {label: float(heads[i]) for i, label in enumerate(node_labels)}, 'flow': {label: float(flow[i]) for i, label in enumerate(link_labels)}}

def todini_pilati(network, solution, mass_tol=1e-6, head_tol=1e-6, feasibility_tol=1e-3, min_gradient=1e-6, flow_tol=1e-8):
    nodes = network.junctions | network.tanks
    link_labels = list(network.pipe_families) + list(network.pumps) + list(network.valves)
    selected_pipes = {label: pipe for label, family in network.pipe_families.items() for pipe in family.options if solution['pipe_status'][pipe.label] == 1}
    tank_heads = {label: tank.head_lower_bound[0] for label, tank in network.tanks.items()}
    hydraulic_solution = {'head': {label: np.empty(network.time_steps + (label in network.tanks)) for label in nodes}, 'flow': {label: np.zeros(network.time_steps) for label in link_labels}, 'pipe_status': solution['pipe_status'].copy()}
    if network.pumps:
        hydraulic_solution['pump_status'] = {label: solution['pump_status'][label].copy() for label in network.pumps}
    if network.valves:
        hydraulic_solution['valve_status'] = {label: solution['valve_status'][label].copy() for label in network.valves}

    for t in range(network.time_steps):
        pump_status = {label: solution['pump_status'][label][t] for label in network.pumps}
        valve_status = {label: solution['valve_status'][label][t] for label in network.valves}
        period_links = {label: (pipe, 'pipe') for label, pipe in selected_pipes.items()} | {label: (pump, 'pump') for label, pump in network.pumps.items() if pump_status[label] == 1} | {label: (valve, 'valve') for label, valve in network.valves.items() if valve_status[label] == 1}
        known_heads = {label: node.head_lower_bound[t] for label, node in network.junctions.items() if node.segment == 'reservoir'} | tank_heads
        period_status, period_solution = _solve_period(nodes, network.junctions, period_links, known_heads, solution, t, mass_tol, head_tol, min_gradient, flow_tol)
        if period_status != 'feasible': return None, period_status, t+1

        tank_inflow = dict.fromkeys(network.tanks, 0.0)
        for label, (obj, component) in period_links.items():
            if obj.start_node.label in tank_inflow: tank_inflow[obj.start_node.label] -= period_solution['flow'][label]
            if obj.end_node.label in tank_inflow: tank_inflow[obj.end_node.label] += period_solution['flow'][label]
        next_head = {label: tank_heads[label] + (tank.tank_coef * tank_inflow[label]) for label, tank in network.tanks.items()}

        for label, head in period_solution['head'].items():
            if head < nodes[label].head_lower_bound[t] - feasibility_tol or head > nodes[label].head_upper_bound[t] + feasibility_tol: return None, 'infeasible', t+1
            hydraulic_solution['head'][label][t] = head
        for label, (obj, component) in period_links.items():
            lower = obj.flow_lower_bound[t] if component == 'pipe' else max(obj.flow_lower_bound[t], obj.positive_flow_lower_bound[t], 0.0)
            if period_solution['flow'][label] < lower - feasibility_tol or period_solution['flow'][label] > obj.flow_upper_bound[t] + feasibility_tol: return None, 'infeasible', t + 1
            hydraulic_solution['flow'][label][t] = period_solution['flow'][label]
        for label, head in next_head.items():
            if head < network.tanks[label].head_lower_bound[t+1] - feasibility_tol or head > network.tanks[label].head_upper_bound[t+1] + feasibility_tol: return None, 'infeasible', t+1
            hydraulic_solution['head'][label][t+1] = head
        tank_heads = next_head

    return hydraulic_solution, 'feasible', network.time_steps
