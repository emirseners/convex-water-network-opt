import igraph as ig

def build_graph(node_indices, links):
    edges = [(node_indices[link.start_node], node_indices[link.end_node]) for link in links]
    return ig.Graph(n=len(node_indices), edges=edges, directed=False)

def fix_bridge_flows(nodes, links):
    nodes, links = list(nodes.values()), list(links.values())
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    graph = build_graph(node_indices, links)
    fixed = {}

    for bridge_id in graph.bridges():
        link = links[bridge_id]

        graph_without_bridge = graph.copy()
        graph_without_bridge.delete_edges([bridge_id])
        membership = graph_without_bridge.connected_components().membership

        start_side = membership[node_indices[link.start_node]]
        end_side = membership[node_indices[link.end_node]]
        start_component = [node for node, side in zip(nodes, membership) if side == start_side]
        end_component = [node for node, side in zip(nodes, membership) if side == end_side]

        if all(getattr(node, 'segment', None) == 'junction' for node in start_component):
            fixed[link] = -sum(node.net_demand for node in start_component)
        elif all(getattr(node, 'segment', None) == 'junction' for node in end_component):
            fixed[link] = sum(node.net_demand for node in end_component)

    for link, flow in fixed.items():
        link.add_flow_bounds(flow, flow)
        if hasattr(link, 'decisions'):
            link.decisions[(link.decisions == -1) & (flow != 0.0)] = 1
            if hasattr(link, 'valve_type'):
                link.decisions[(link.decisions == -1) & (flow == 0.0)] = 0
