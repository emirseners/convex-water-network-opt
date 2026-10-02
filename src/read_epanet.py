from __future__ import annotations
from .helpers import round_bound, write_bounds
import numpy as np
import wntr

def print_water_network_info(water_network_path):
    water_network = wntr.network.WaterNetworkModel(water_network_path)

    print(f'Duration: {water_network.options.time.duration} seconds')
    print(f'Hydraulic timestep: {water_network.options.time.hydraulic_timestep} seconds')
    print(f'Pattern timestep: {water_network.options.time.pattern_timestep} seconds')

    print(f'Number of junctions: {water_network.num_junctions}')
    print(f'Number of reservoirs: {water_network.num_reservoirs}')
    print(f'Number of tanks: {water_network.num_tanks}')
    print(f'Number of pipes: {water_network.num_pipes}')
    print(f'Number of pumps: {water_network.num_pumps}')
    print(f'Number of valves: {water_network.num_valves}')
    print(f'Number of patterns: {water_network.num_patterns}')
    print(f'Number of curves: {water_network.num_curves}')

    print('Pump types: ', [each_pump.pump_type for _, each_pump in water_network.pumps()])
    print('Valve types: ', [each_valve.valve_type for _, each_valve in water_network.valves()])
    print('Number of pipes with check valves: ', sum(each_pipe.check_valve for _, each_pipe in water_network.pipes()))

class Junction:
    def __init__(self, label: str, segment: str, elevation: np.array, net_demand: np.array):
        self.label = label
        self.segment = segment
        self.net_demand = net_demand
        self.elevation = elevation
        self.incoming_edges = []
        self.outgoing_edges = []

    def add_incoming_edge(self, incoming_edge: Pipe | Valve | Pump):
        self.incoming_edges.append(incoming_edge)

    def add_outgoing_edge(self, outgoing_edge: Pipe | Valve | Pump):
        self.outgoing_edges.append(outgoing_edge)

    def add_head_bounds(self, head_lower_bound=None, head_upper_bound=None):
        if head_lower_bound is not None:
            self.head_lower_bound = np.array(head_lower_bound, dtype=float) if not hasattr(self, 'head_lower_bound') else np.maximum(head_lower_bound, self.head_lower_bound)
        if head_upper_bound is not None:
            self.head_upper_bound = np.array(head_upper_bound, dtype=float) if not hasattr(self, 'head_upper_bound') else np.minimum(head_upper_bound, self.head_upper_bound)

class Tank:
    def __init__(self, label: str, elevation: float, diameter: float, area: float):
        self.label = label
        self.elevation = elevation
        self.diameter = diameter
        self.area = area
        self.incoming_edges = []
        self.outgoing_edges = []

    def add_incoming_edge(self, incoming_edge: Pipe | Valve | Pump):
        self.incoming_edges.append(incoming_edge)

    def add_outgoing_edge(self, outgoing_edge: Pipe | Valve | Pump):
        self.outgoing_edges.append(outgoing_edge)

    def add_head_bounds(self, head_lower_bound=None, head_upper_bound=None):
        if head_lower_bound is not None:
            if not hasattr(self, 'head_lower_bound'):
                self.head_lower_bound = head_lower_bound
            elif head_lower_bound.shape == self.head_lower_bound[:-1].shape:
                self.head_lower_bound[:-1] = np.maximum(head_lower_bound, self.head_lower_bound[:-1])
            else:
                self.head_lower_bound = np.maximum(head_lower_bound, self.head_lower_bound)

        if head_upper_bound is not None:
            if not hasattr(self, 'head_upper_bound'):
                self.head_upper_bound = head_upper_bound
            elif head_upper_bound.shape == self.head_upper_bound[:-1].shape:
                self.head_upper_bound[:-1] = np.minimum(head_upper_bound, self.head_upper_bound[:-1])
            else:
                self.head_upper_bound = np.minimum(head_upper_bound, self.head_upper_bound)

class Pipe:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, length: float, diameter: float, roughness: float, loss_coefficient: float, exponent: float):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.loss_coefficient = loss_coefficient
        self.exponent = exponent

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.array(flow_lower_bound, dtype=float) if not hasattr(self, 'flow_lower_bound') else np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.array(flow_upper_bound, dtype=float) if not hasattr(self, 'flow_upper_bound') else np.minimum(flow_upper_bound, self.flow_upper_bound)

class Valve:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, length: float, diameter: float, roughness: float, loss_coefficient: float, exponent: float, time_steps: int):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.loss_coefficient = loss_coefficient
        self.exponent = exponent
        self.decisions = np.full(time_steps, -1, dtype=int)

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.array(flow_lower_bound, dtype=float) if not hasattr(self, 'flow_lower_bound') else np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.array(flow_upper_bound, dtype=float) if not hasattr(self, 'flow_upper_bound') else np.minimum(flow_upper_bound, self.flow_upper_bound)

class Pump:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, time_steps: int, pump_type: str, fixed_cost: np.array, operational_cost: np.array):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.pump_type = pump_type
        self.fixed_cost = fixed_cost
        self.operational_cost = operational_cost
        self.decisions = np.full(time_steps, -1, dtype=int)

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.array(flow_lower_bound, dtype=float) if not hasattr(self, 'flow_lower_bound') else np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.array(flow_upper_bound, dtype=float) if not hasattr(self, 'flow_upper_bound') else np.minimum(flow_upper_bound, self.flow_upper_bound)

    def add_pump_costs(self, fixed_cost: np.array, operational_cost: np.array):
        self.fixed_cost = fixed_cost
        self.operational_cost = operational_cost

    #We only consider HEAD pumps. POWER pumps and variable speed pumps are neglected
    def add_head_curve(self, curve_a, curve_b, curve_c):
        self.curve_a, self.curve_b, self.curve_c = curve_a, curve_b, curve_c

def fetch_epanet_data(water_network_path, head_loss_equation, flow_scale, head_scale):
    water_network = wntr.network.WaterNetworkModel(water_network_path)

    patterns = {}
    junctions = {}
    tanks = {}
    pumps = {}
    valves = {}
    pipes = {}

    duration = int(water_network.options.time.duration)
    hydraulic_step_length = int(water_network.options.time.hydraulic_timestep)
    pattern_step_length = int(water_network.options.time.pattern_timestep)
    pattern_start = int(water_network.options.time.pattern_start)
    time_steps = int(duration / hydraulic_step_length)

    def expand_pattern(multipliers: np.array, base: float) -> np.array:
        multipliers = np.asarray(multipliers, dtype=float)

        if multipliers.size == 0:
            return np.full(time_steps, base, dtype=float)

        sample_times = (pattern_start + np.arange(time_steps) * hydraulic_step_length)
        pattern_indices = (sample_times // pattern_step_length).astype(int) % multipliers.size

        return base * multipliers[pattern_indices]

    global_energy_price = water_network.options.energy.global_price
    global_energy_price_pattern = water_network.options.energy.global_pattern
    global_demand_pattern = water_network.options.hydraulic.pattern
    global_demand_multiplier = water_network.options.hydraulic.demand_multiplier

    for pattern_name, each_pattern in water_network.patterns():
        patterns[pattern_name] = each_pattern.multipliers

    for reservoir_name, each_reservoir in water_network.reservoirs():
        reservoir_elevation = expand_pattern(patterns[each_reservoir.head_pattern_name], each_reservoir.base_head) if each_reservoir.head_pattern_name in patterns else np.repeat(each_reservoir.base_head, time_steps)
        junctions[reservoir_name] = Junction(reservoir_name, 'reservoir', head_scale * reservoir_elevation, None)
        junctions[reservoir_name].add_head_bounds(head_scale * reservoir_elevation, head_scale * reservoir_elevation)

    for tank_name, each_tank in water_network.tanks():
        diameter = each_tank.diameter
        area = (flow_scale / head_scale) * round(((diameter / 2) ** 2) * np.pi, 4)
        head_lower_bound = each_tank.elevation + np.insert(np.repeat(each_tank.min_level, time_steps), 0, each_tank.init_level)
        head_upper_bound = each_tank.elevation + np.insert(np.repeat(each_tank.max_level, time_steps), 0, each_tank.init_level)
        head_lower_bound[-1] = max(head_lower_bound[-1], each_tank.elevation + each_tank.init_level)
        tanks[tank_name] = Tank(tank_name, head_scale * each_tank.elevation, diameter, area)
        tanks[tank_name].add_head_bounds(head_scale * head_lower_bound, head_scale * head_upper_bound)

    for junction_name, each_junction in water_network.junctions():
        junction_demand_pattern = each_junction.demand_pattern if each_junction.demand_pattern in patterns else global_demand_pattern
        net_demand = expand_pattern(patterns[junction_demand_pattern], each_junction.base_demand) if junction_demand_pattern is not None else np.repeat(each_junction.base_demand, time_steps)
        junction_elevation = head_scale * np.repeat(each_junction.elevation, time_steps)
        junctions[junction_name] = Junction(junction_name, 'junction', junction_elevation, net_demand * global_demand_multiplier * flow_scale)

    nodes = junctions | tanks

    for pump_name, each_pump in water_network.pumps():
        base_energy_price = each_pump.energy_price or global_energy_price or 0.001
        energy_price_pattern = each_pump.energy_pattern or global_energy_price_pattern
        energy_price = expand_pattern(patterns[energy_price_pattern], base_energy_price) if energy_price_pattern is not None else np.repeat(base_energy_price, time_steps)
        price_coefficient = hydraulic_step_length * energy_price

        start_node = nodes[each_pump.start_node_name]
        end_node = nodes[each_pump.end_node_name]
        pumps[pump_name] = Pump(pump_name, start_node, end_node, time_steps, each_pump.pump_type, 54 * price_coefficient, (196 * price_coefficient) / flow_scale)

        start_node.add_outgoing_edge(pumps[pump_name])
        end_node.add_incoming_edge(pumps[pump_name])

        curve_a, curve_b, curve_c = each_pump.get_head_curve_coefficients()
        curve_c = curve_c if abs(curve_c - round(curve_c)) > 1e-4 else round(curve_c)
        pumps[pump_name].add_head_curve(curve_a * head_scale, -((curve_b * head_scale) / (flow_scale ** curve_c)), curve_c)

    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for pipe_name, each_pipe in water_network.pipes():
        length = each_pipe.length
        diameter = each_pipe.diameter
        roughness = each_pipe.roughness
        check_valve = each_pipe.check_valve

        start_node = nodes[each_pipe.start_node_name]
        end_node = nodes[each_pipe.end_node_name]

        # Currently, all input files have HW. If DW, then one should adjust coefficients with adequate units
        if head_loss_equation == 'HW':
            loss_coefficient = (head_scale * ((10.66682948893005 * length) / ((roughness ** 1.852) * (diameter ** 4.871)))) / (flow_scale ** 1.852)
        else: # Reynolds dependent roughness coeff? Currently, we use Colebrook approximation
            loss_coefficient = (head_scale * ((8 * length * ((2 * np.log10((3.7 * diameter) / roughness)) ** (-2))) / ((np.pi ** 2) * 9.80665 * (diameter ** 5))))  / (flow_scale ** 2)

        if not check_valve:
            pipes[pipe_name] = Pipe(pipe_name, start_node, end_node, length, diameter, roughness, loss_coefficient, exponent)
            start_node.add_outgoing_edge(pipes[pipe_name])
            end_node.add_incoming_edge(pipes[pipe_name])
        else: # Currently, no instance exists with valves other than check valves, thus we only consider them
            valves[pipe_name] = Valve(pipe_name, start_node, end_node, length, diameter, roughness, loss_coefficient, exponent, time_steps)
            start_node.add_outgoing_edge(valves[pipe_name])
            end_node.add_incoming_edge(valves[pipe_name])

    return junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_step_length

def get_operational_head_bounds(node):
    if isinstance(node, Tank):
        return node.head_lower_bound[:-1], node.head_upper_bound[:-1]
    return node.head_lower_bound, node.head_upper_bound

def propagate_flow(edges):
    for edge in edges.values():
        start_head_lower_bound, start_head_upper_bound = get_operational_head_bounds(edge.start_node)
        end_head_lower_bound, end_head_upper_bound = get_operational_head_bounds(edge.end_node)
        head_diff_lb = start_head_lower_bound - end_head_upper_bound
        head_diff_ub = start_head_upper_bound - end_head_lower_bound
        if isinstance(edge, Pipe):
            edge.add_flow_bounds(round_bound(np.sign(head_diff_lb) * ((np.abs(head_diff_lb) / edge.loss_coefficient) ** (1 / edge.exponent)), 'floor'), round_bound(np.sign(head_diff_ub) * ((np.abs(head_diff_ub) / edge.loss_coefficient) ** (1 / edge.exponent)), 'ceiling'))
        elif isinstance(edge, Valve):
            edge.add_flow_bounds(round_bound(np.where(edge.decisions == 1, (np.maximum(head_diff_lb, 0.0) / edge.loss_coefficient) ** (1 / edge.exponent), 0.0), 'floor'), round_bound(np.where(edge.decisions == 0, 0.0, (np.maximum(head_diff_ub, 0.0) / edge.loss_coefficient) ** (1 / edge.exponent)), 'ceiling'))
        elif isinstance(edge, Pump):
            edge.add_flow_bounds(round_bound(np.where(edge.decisions == 1, (np.maximum(head_diff_lb + edge.curve_a, 0.0) / -edge.curve_b) ** (1 / edge.curve_c), 0.0), 'floor'), round_bound(np.where(edge.decisions == 0, 0.0, (np.maximum(head_diff_ub + edge.curve_a, 0.0) / -edge.curve_b) ** (1 / edge.curve_c)), 'ceiling'))

def head_loss(edge, flow):
    return edge.loss_coefficient * np.sign(flow) * (np.abs(flow) ** edge.exponent)

def propagate_head(nodes):
    for node in nodes.values():
        for edge in node.incoming_edges:
            start_head_lower_bound, start_head_upper_bound = get_operational_head_bounds(edge.start_node)
            if isinstance(edge, Pipe):
                node.add_head_bounds(round_bound(start_head_lower_bound - head_loss(edge, edge.flow_upper_bound), 'floor'), round_bound(start_head_upper_bound - head_loss(edge, edge.flow_lower_bound), 'ceiling'))
            elif isinstance(edge, Valve):
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, start_head_lower_bound - head_loss(edge, edge.flow_upper_bound), -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, start_head_upper_bound - head_loss(edge, edge.flow_lower_bound), np.inf), 'ceiling'))
            elif isinstance(edge, Pump):
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, start_head_lower_bound + edge.curve_a + (edge.curve_b * (edge.flow_upper_bound ** edge.curve_c)), -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, start_head_upper_bound + edge.curve_a + (edge.curve_b * (edge.flow_lower_bound ** edge.curve_c)), np.inf), 'ceiling'))

        for edge in node.outgoing_edges:
            end_head_lower_bound, end_head_upper_bound = get_operational_head_bounds(edge.end_node)
            if isinstance(edge, Pipe):
                node.add_head_bounds(round_bound(end_head_lower_bound + head_loss(edge, edge.flow_lower_bound), 'floor'), round_bound(end_head_upper_bound + head_loss(edge, edge.flow_upper_bound), 'ceiling'))
            elif isinstance(edge, Valve):
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, end_head_lower_bound + head_loss(edge, edge.flow_lower_bound), -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, end_head_upper_bound + head_loss(edge, edge.flow_upper_bound), np.inf), 'ceiling'))
            elif isinstance(edge, Pump):
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, end_head_lower_bound - (edge.curve_a + (edge.curve_b * (edge.flow_lower_bound ** edge.curve_c))), -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, end_head_upper_bound - (edge.curve_a + (edge.curve_b * (edge.flow_upper_bound ** edge.curve_c))), np.inf), 'ceiling'))

def infere_decisions(pumps, valves, time_steps, epsilon=0.1):
    for pump in pumps.values():
        for t in range(time_steps):
            if pump.decisions[t] != -1: continue
            if pump.flow_lower_bound[t] >= epsilon:
                pump.decisions[t] = 1

    for valve in valves.values():
        for t in range(time_steps):
            if valve.decisions[t] != -1: continue
            if valve.flow_upper_bound[t] == 0.0:
                valve.decisions[t] = 0
            if valve.flow_lower_bound[t] >= epsilon:
                valve.decisions[t] = 1

def preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, post_obbt, results_directory):
    for r in range(5):
        propagate_flow(pipes | pumps | valves)
        propagate_head(junctions | tanks)
        infere_decisions(pumps, valves, time_steps)
        write_bounds(junctions, tanks, pipes, pumps, valves, time_steps, results_directory, sheet_name = f'bounds_post_obbt_{r}' if post_obbt else f'bounds_pre_obbt_{r}', output_file = 'decision_variable_bounds.xlsx')

def set_bounds(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, results_directory, head_scale, pressure_margin=100):
    for pump in pumps.values():
        flow_upper_bound = (-pump.curve_a / pump.curve_b) ** (1 / pump.curve_c)
        pump.add_flow_bounds(np.zeros(time_steps), np.full(time_steps, flow_upper_bound))

    for valve in valves.values():
        valve.add_flow_bounds(np.zeros(time_steps))

    '''flow_capacity = np.array([np.maximum(junction.net_demand, 0) for junction in junctions.values() if junction.net_demand is not None]).sum(axis=0) + (np.sum([tank.area * (tank.head_upper_bound[1:] - tank.head_lower_bound[1:]) for tank in tanks.values()], axis=0) / hydraulic_timestep)
    for link in (pipes | pumps | valves).values():
        link.add_flow_bounds(-flow_capacity, flow_capacity)'''

    global_head_ub = np.max([junction.elevation for junction in junctions.values()] + [tank.head_upper_bound[:-1] for tank in tanks.values()], axis=0) + sum((pump.curve_a for pump in pumps.values()), 0.0)

    for junction in junctions.values():
        if junction.segment == 'reservoir': continue
        #junction.add_head_bounds(junction.elevation if min(junction.net_demand) > 0 else junction.elevation - (head_scale * pressure_margin), global_head_ub)
        junction.add_head_bounds(np.maximum(junction.elevation - (head_scale * pressure_margin), 0.0), global_head_ub + (head_scale * pressure_margin))

    preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, False, results_directory)

def set_timesteps(junctions, tanks, pipes, pumps, valves, time_steps):
    for junction in junctions.values():
        if junction.net_demand is not None:
            junction.net_demand = junction.net_demand[time_steps]
        junction.elevation = junction.elevation[time_steps]
        junction.head_lower_bound = junction.head_lower_bound[time_steps]
        junction.head_upper_bound = junction.head_upper_bound[time_steps]

    for tank in tanks.values():
        tank.head_lower_bound = tank.head_lower_bound[time_steps + [time_steps[-1] + 1]]
        tank.head_upper_bound = tank.head_upper_bound[time_steps + [time_steps[-1] + 1]]

    for pipe in pipes.values():
        pipe.flow_lower_bound = pipe.flow_lower_bound[time_steps]
        pipe.flow_upper_bound = pipe.flow_upper_bound[time_steps]

    for valve in valves.values():
        valve.flow_lower_bound = valve.flow_lower_bound[time_steps]
        valve.flow_upper_bound = valve.flow_upper_bound[time_steps]
        valve.decisions = valve.decisions[time_steps]

    for pump in pumps.values():
        pump.fixed_cost = pump.fixed_cost[time_steps]
        pump.operational_cost = pump.operational_cost[time_steps]
        pump.flow_lower_bound = pump.flow_lower_bound[time_steps]
        pump.flow_upper_bound = pump.flow_upper_bound[time_steps]
        pump.decisions = pump.decisions[time_steps]

    return junctions, tanks, pipes, pumps, valves
