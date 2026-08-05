from __future__ import annotations
from helpers import round_bound, write_bounds
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
    def __init__(self, label: str, elevation: float, diameter: float, area: float, storage_initial: float, storage_lower_bound: np.array, storage_upper_bound: np.array):
        self.label = label
        self.elevation = elevation
        self.diameter = diameter
        self.area = area
        self.storage_initial = storage_initial
        self.storage_lower_bound = storage_lower_bound
        self.storage_upper_bound = storage_upper_bound
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

class Pipe:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, length: float, diameter: float, roughness: float, loss_coefficient: float, check_valve: bool):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.check_valve = check_valve
        self.loss_coefficient = loss_coefficient

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None, pass_valve=True):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.array(flow_lower_bound, dtype=float) if not hasattr(self, 'flow_lower_bound') else np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.array(flow_upper_bound, dtype=float) if not hasattr(self, 'flow_upper_bound') else np.minimum(flow_upper_bound, self.flow_upper_bound)
        if self.check_valve and pass_valve:
            self.corresponding_check_valve.add_flow_bounds(flow_lower_bound, flow_upper_bound, False)

class Valve:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, diameter: float, time_steps: int, valve_type: str):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.diameter = diameter
        self.valve_type = valve_type
        self.decisions = np.full(time_steps, -1, dtype=int)

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None, pass_pipe=True):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.array(flow_lower_bound, dtype=float) if not hasattr(self, 'flow_lower_bound') else np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.array(flow_upper_bound, dtype=float) if not hasattr(self, 'flow_upper_bound') else np.minimum(flow_upper_bound, self.flow_upper_bound)
        if self.valve_type == 'check valve' and pass_pipe:
            self.corresponding_pipe.add_flow_bounds(flow_lower_bound, flow_upper_bound, False)

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
        self.curve_a, self.curve_b, self.curve_c = curve_a, curve_b, curve_c if abs(curve_c - 1.0) > 1e-6 else 1

def fetch_epanet_data(water_network_path, head_loss_equation):
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
        junctions[reservoir_name] = Junction(reservoir_name, 'reservoir', reservoir_elevation, None)
        junctions[reservoir_name].add_head_bounds(reservoir_elevation, reservoir_elevation)

    for tank_name, each_tank in water_network.tanks():
        diameter = each_tank.diameter
        area = ((diameter/2) ** 2) * np.pi
        storage_initial = each_tank.init_level * area
        storage_lower_bound = np.insert(np.repeat(each_tank.min_level * area, time_steps), 0, storage_initial)
        storage_upper_bound = np.insert(np.repeat(each_tank.max_level * area, time_steps), 0, storage_initial)
        tanks[tank_name] = Tank(tank_name, each_tank.elevation, diameter, area, storage_initial, storage_lower_bound, storage_upper_bound)

    for junction_name, each_junction in water_network.junctions():
        junction_demand_pattern = each_junction.demand_pattern if each_junction.demand_pattern in patterns else global_demand_pattern
        net_demand = expand_pattern(patterns[junction_demand_pattern], each_junction.base_demand) if junction_demand_pattern is not None else np.repeat(each_junction.base_demand, time_steps)
        junction_elevation = np.repeat(each_junction.elevation, time_steps)
        junctions[junction_name] = Junction(junction_name, 'junction', junction_elevation, net_demand * global_demand_multiplier)

    nodes = junctions | tanks

    for pump_name, each_pump in water_network.pumps():
        base_energy_price = each_pump.energy_price or global_energy_price or 0.001
        energy_price_pattern = each_pump.energy_pattern or global_energy_price_pattern
        energy_price = expand_pattern(patterns[energy_price_pattern], base_energy_price) if energy_price_pattern is not None else np.repeat(base_energy_price, time_steps)
        price_coefficient = hydraulic_step_length * energy_price

        start_node = nodes[each_pump.start_node_name]
        end_node = nodes[each_pump.end_node_name]
        pumps[pump_name] = Pump(pump_name, start_node, end_node, time_steps, each_pump.pump_type, 54 * price_coefficient, 196 * price_coefficient)

        start_node.add_outgoing_edge(pumps[pump_name])
        end_node.add_incoming_edge(pumps[pump_name])

        curve_a, curve_b, curve_c = each_pump.get_head_curve_coefficients()
        pumps[pump_name].add_head_curve(curve_a, -curve_b, curve_c)

    for pipe_name, each_pipe in water_network.pipes():
        length = each_pipe.length
        diameter = each_pipe.diameter
        roughness = each_pipe.roughness
        check_valve = each_pipe.check_valve

        start_node = nodes[each_pipe.start_node_name]
        end_node = nodes[each_pipe.end_node_name]

        # Currently, all input files have HW. If DW, then one should adjust coefficients with adequate units
        if head_loss_equation == 'HW':
            loss_coefficient = (10.67 * length) / ((roughness ** 1.852) * (diameter ** 4.8704))
        else: #Reynolds dependent roughness coeff? Currently, we use Colebrook approximation
            loss_coefficient = (8 * length * ((2 * np.log10((3.7 * diameter) / roughness)) ** (-2))) / ((np.pi ** 2) * 9.80665 * (diameter ** 5))

        pipes[pipe_name] = Pipe(pipe_name, start_node, end_node, length, diameter, roughness, loss_coefficient, check_valve)
        start_node.add_outgoing_edge(pipes[pipe_name])
        end_node.add_incoming_edge(pipes[pipe_name])

    for valve_name, each_valve in water_network.valves():
        diameter = each_valve.diameter
        valve_type = each_valve.valve_type

        start_node = nodes[each_valve.start_node_name]
        end_node = nodes[each_valve.end_node_name]

        valves[valve_name] = Valve(valve_name, start_node, end_node, diameter, time_steps, valve_type)
        start_node.add_outgoing_edge(valves[valve_name])
        end_node.add_incoming_edge(valves[valve_name])

    for pipe_name, pipe in pipes.items():
        if not pipe.check_valve: continue

        original_end_node = pipe.end_node
        elevation = original_end_node.elevation if isinstance(original_end_node, Junction) else np.full(time_steps, original_end_node.elevation)

        auxiliary_junction = Junction(f'AUX_J_{pipe_name}', 'junction', elevation, np.zeros(time_steps))
        junctions[f'AUX_J_{pipe_name}'] = auxiliary_junction

        pipe.end_node = auxiliary_junction
        auxiliary_junction.add_incoming_edge(pipe)
        original_end_node.incoming_edges.remove(pipe)

        valves[f'AUX_V_{pipe_name}'] = Valve(f'AUX_V_{pipe_name}', auxiliary_junction, original_end_node, pipe.diameter, time_steps, 'check valve')
        auxiliary_junction.add_outgoing_edge(valves[f'AUX_V_{pipe_name}'])
        original_end_node.add_incoming_edge(valves[f'AUX_V_{pipe_name}'])

        pipe.corresponding_check_valve = valves[f'AUX_V_{pipe_name}']
        valves[f'AUX_V_{pipe_name}'].corresponding_pipe = pipe

    return junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_step_length

def propagate_flow(edges, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for edge in edges.values():
        head_diff_lb = edge.start_node.head_lower_bound - edge.end_node.head_upper_bound
        head_diff_ub = edge.start_node.head_upper_bound - edge.end_node.head_lower_bound
        if isinstance(edge, Pipe):
            edge.add_flow_bounds(round_bound(np.sign(head_diff_lb) * ((np.abs(head_diff_lb) / edge.loss_coefficient) ** (1 / exponent)), 'floor'), round_bound(np.sign(head_diff_ub) * ((np.abs(head_diff_ub) / edge.loss_coefficient) ** (1 / exponent)), 'ceiling'))
        elif isinstance(edge, Pump):
            edge.add_flow_bounds(round_bound(np.where(edge.decisions == 1, (np.maximum(head_diff_lb + edge.curve_a, 0.0) / -edge.curve_b) ** (1 / edge.curve_c), -np.inf), 'floor'), round_bound((np.maximum(head_diff_ub + edge.curve_a, 0.0) / -edge.curve_b) ** (1 / edge.curve_c), 'ceiling'))

def head_loss(edge, flow, exponent):
    return edge.loss_coefficient * np.sign(flow) * (np.abs(flow) ** exponent)

def propagate_head(nodes, head_loss_equation):
    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for node in nodes.values():
        for edge in node.incoming_edges:
            if isinstance(edge, Pipe):
                node.add_head_bounds(round_bound(edge.start_node.head_lower_bound - head_loss(edge, edge.flow_upper_bound, exponent), 'floor'), round_bound(edge.start_node.head_upper_bound - head_loss(edge, edge.flow_lower_bound, exponent), 'ceiling'))
            elif isinstance(edge, Valve):
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, edge.start_node.head_lower_bound, -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, edge.start_node.head_upper_bound, np.inf), 'ceiling'))
            else:
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, edge.start_node.head_lower_bound + edge.curve_a + (edge.curve_b * (edge.flow_upper_bound ** edge.curve_c)), -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, edge.start_node.head_upper_bound + edge.curve_a + (edge.curve_b * (edge.flow_lower_bound ** edge.curve_c)), np.inf), 'ceiling'))

        for edge in node.outgoing_edges:
            if isinstance(edge, Pipe):
                node.add_head_bounds(round_bound(edge.end_node.head_lower_bound + head_loss(edge, edge.flow_lower_bound, exponent), 'floor'), round_bound(edge.end_node.head_upper_bound + head_loss(edge, edge.flow_upper_bound, exponent), 'ceiling'))
            elif isinstance(edge, Valve):
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, edge.end_node.head_lower_bound, -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, edge.end_node.head_upper_bound, np.inf), 'ceiling'))
            else:
                node.add_head_bounds(round_bound(np.where(edge.decisions == 1, edge.end_node.head_lower_bound - (edge.curve_a + (edge.curve_b * (edge.flow_lower_bound ** edge.curve_c))), -np.inf), 'floor'), round_bound(np.where(edge.decisions == 1, edge.end_node.head_upper_bound - (edge.curve_a + (edge.curve_b * (edge.flow_upper_bound ** edge.curve_c))), np.inf), 'ceiling'))

def infere_decisions(pumps, valves, time_steps, epsilon=1e-4):
    for pump in pumps.values():
        for t in range(time_steps):
            if pump.decisions[t] != -1: continue
            if pump.flow_lower_bound[t] >= epsilon:
                pump.decisions[t] = 1

    for valve in valves.values():
        for t in range(time_steps):
            if valve.decisions[t] != -1: continue
            if valve.flow_upper_bound[t] <= -epsilon or valve.flow_lower_bound[t] >= epsilon:
                valve.decisions[t] = 1

def preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, head_loss_equation, post_obbt, results_directory):
    for r in range(5):
        propagate_flow(pipes | pumps, head_loss_equation)
        propagate_head(junctions | tanks, head_loss_equation)
        infere_decisions(pumps, valves, time_steps)
        write_bounds(junctions, tanks, pipes, pumps, valves, time_steps, results_directory, sheet_name = f'bounds_post_obbt_{r}' if post_obbt else f'bounds_pre_obbt_{r}', output_file = 'decision_variable_bounds.xlsx')

def set_bounds(junctions, tanks, pipes, pumps, valves, head_loss_equation, time_steps, hydraulic_timestep, results_directory, pressure_margin = 100):
    for pump in pumps.values():
        flow_upper_bound = (-pump.curve_a / pump.curve_b) ** (1 / pump.curve_c)
        pump.add_flow_bounds(np.zeros(time_steps), np.full(time_steps, flow_upper_bound))

    for tank in tanks.values():
        tank.add_head_bounds(tank.elevation + (tank.storage_lower_bound[:-1] / tank.area), tank.elevation + (tank.storage_upper_bound[:-1] / tank.area))

    '''flow_capacity = np.array([np.maximum(junction.net_demand, 0) for junction in junctions.values() if junction.net_demand is not None]).sum(axis=0) + (np.sum([tank.storage_upper_bound[1:] - tank.storage_lower_bound[1:] for tank in tanks.values()], axis=0) / hydraulic_timestep)
    for link in (pipes | pumps | valves).values():
        link.add_flow_bounds(-flow_capacity, flow_capacity)'''

    global_head_ub = np.max([junction.elevation for junction in junctions.values()] + [tank.head_upper_bound for tank in tanks.values()], axis=0) + sum((pump.curve_a for pump in pumps.values()), 0.0)

    for junction in junctions.values():
        if junction.segment == 'reservoir': continue
        #junction.add_head_bounds(junction.elevation if min(junction.net_demand) > 0 else junction.elevation - pressure_margin, global_head_ub)
        junction.add_head_bounds(junction.elevation - pressure_margin, global_head_ub + pressure_margin)

    for pipe in pipes.values():
        if not pipe.check_valve: continue
        pipe.add_flow_bounds(np.zeros(time_steps))

    preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, head_loss_equation, False, results_directory)

def set_timesteps(junctions, tanks, pipes, pumps, valves, time_steps):
    for junction in junctions.values():
        if junction.net_demand is not None:
            junction.net_demand = junction.net_demand[time_steps]
        junction.elevation = junction.elevation[time_steps]
        junction.head_lower_bound = junction.head_lower_bound[time_steps]
        junction.head_upper_bound = junction.head_upper_bound[time_steps]

    for tank in tanks.values():
        tank.head_lower_bound = tank.head_lower_bound[time_steps]
        tank.head_upper_bound = tank.head_upper_bound[time_steps]
        tank.storage_lower_bound = tank.storage_lower_bound[time_steps + [time_steps[-1] + 1]]
        tank.storage_upper_bound = tank.storage_upper_bound[time_steps + [time_steps[-1] + 1]]

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
