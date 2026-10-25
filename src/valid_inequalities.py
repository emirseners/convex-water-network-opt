import numpy as np
import igraph as ig
from itertools import combinations
from .helpers import valid_inequality_key
from .read_epanet import Pipe, PipeFamily, Pump, Tank

def build_graph(node_indices, links):
    edges = [(node_indices[link.start_node], node_indices[link.end_node]) for link in links]
    return ig.Graph(n=len(node_indices), edges=edges, directed=False)

def fix_bridge_flows(network):
    nodes = list(network.junctions.values()) + list(network.tanks.values())
    links = list(network.pipe_families.values()) + list(network.pumps.values()) + list(network.valves.values())
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    graph = build_graph(node_indices, links)
    fixed = {}

    for bridge_id in graph.bridges():
        link = links[bridge_id]

        graph_without_bridge = graph.copy()
        graph_without_bridge.delete_edges([bridge_id])
        membership = graph_without_bridge.connected_components().membership

        start_component = [node for node, side in zip(nodes, membership) if side == membership[node_indices[link.start_node]]]
        end_component = [node for node, side in zip(nodes, membership) if side == membership[node_indices[link.end_node]]]

        if all(getattr(node, 'segment', None) == 'junction' for node in start_component):
            fixed[link] = -sum(node.net_demand for node in start_component)
        elif all(getattr(node, 'segment', None) == 'junction' for node in end_component):
            fixed[link] = sum(node.net_demand for node in end_component)

    for link, flow in fixed.items():
        if not isinstance(link, PipeFamily):
            link.add_flow_bounds(flow, flow)
            continue

        for pipe in link.options:
            if pipe.selection_decision == 0: continue

            if pipe.selection_decision == 1:
                pipe.add_flow_bounds(flow, flow)
            else:
                pipe.add_flow_bounds(np.minimum(flow, 0.0), np.maximum(flow, 0.0))
                pipe.add_directional_flow_bounds(np.maximum(flow, 0.0), np.minimum(flow, 0.0))

                if np.any((flow < pipe.flow_lower_bound) | (flow > pipe.flow_upper_bound)):
                    pipe.selection_decision = 0
                    pipe.add_flow_bounds(0.0, 0.0)

def net_demand_bounds(nodes, start, end):
    demand_lower = demand_upper = 0.0

    for node in nodes:
        if isinstance(node, Tank):
            demand_lower += (node.head_lower_bound[end] - node.head_upper_bound[start]) / node.tank_coef
            demand_upper += (node.head_upper_bound[end] - node.head_lower_bound[start]) / node.tank_coef
        else:
            demand = float(node.net_demand[start:end].sum())
            demand_lower += demand
            demand_upper += demand

    return demand_lower, demand_upper

def aggregated_flow_bound_inequalities(nodes, links, horizons):
    node_set = set(nodes)
    boundary = [(edge, edge.end_node in node_set) for edge in links if (edge.start_node in node_set) != (edge.end_node in node_set)]
    inequalities = set()

    for start, end in horizons:
        demand_lower, demand_upper = net_demand_bounds(nodes, start, end)

        for orientation in (1, -1):
            rhs = demand_lower if orientation == 1 else -demand_upper
            terms = []

            for edge, direction in boundary:
                forward = direction if orientation == 1 else not direction
                variable = 'pipe_status' if isinstance(edge, Pipe) else 'pump_status' if isinstance(edge, Pump) else 'valve_status'

                for period in range(start, end):
                    lower = float(edge.flow_lower_bound[period])
                    upper = float(edge.flow_upper_bound[period])
                    capacity = upper if forward else -lower

                    if capacity == 0.0: continue

                    if isinstance(edge, Pipe):
                        if lower < 0.0 < upper:
                            coefficient = capacity if forward else -capacity
                            terms.append((variable, edge.label, period, coefficient))

                            if not forward:
                                rhs -= capacity
                        else:
                            rhs -= capacity

                    elif edge.status_decisions[period] == 0:
                        continue
                    elif edge.status_decisions[period] == 1 or lower == upper:
                        rhs -= capacity
                    else:
                        terms.append((variable, edge.label, period, capacity))

            if terms:
                inequalities.add(valid_inequality_key(terms, 'ge', rhs))

    return inequalities

def is_connected(graph, node_indices, subset):
    return graph.induced_subgraph([node_indices[node] for node in subset]).is_connected()

#will be adjusted later
def node_subset(nodes, links, horizons, max_cardinality=3):
    nodes, links = list(nodes.values()), list(links.values())
    non_reservoir_nodes = [node for node in nodes if isinstance(node, Tank) or node.segment != 'reservoir']
    node_indices = {node: index for index, node in enumerate(nodes)}
    graph = build_graph(node_indices, links)
    inequalities = set()

    for cardinality in range(1, min(max_cardinality, len(non_reservoir_nodes)) + 1):
        for subset in combinations(non_reservoir_nodes, cardinality):
            if is_connected(graph, node_indices, subset):
                inequalities.update(aggregated_flow_bound_inequalities(subset, links, horizons))

    return inequalities