from __future__ import annotations
import numpy as np
import math
import wntr

def print_water_network_info(water_network_path):
    water_network = wntr.network.WaterNetworkModel(water_network_path)

    print(f'Duration: {water_network.options.time.duration} seconds')
    print(f'Hydraulic timestep: {water_network.options.time.hydraulic_step_length} seconds')
    print(f'Pattern timestep: {water_network.options.time.pattern_step_length} seconds')
    
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

    def add_head_bounds(self, head_lower_bound: np.array, head_upper_bound: np.array):
        self.head_lower_bound = head_lower_bound
        self.head_upper_bound = head_upper_bound

class Tank:
    def __init__(self, label: str, elevation: float, diameter: float, area: float, storage_initial: float, storage_lower_bound: float, storage_upper_bound: float):
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

    def add_head_bounds(self, head_lower_bound: np.array, head_upper_bound: np.array):
        self.head_lower_bound = head_lower_bound
        self.head_upper_bound = head_upper_bound

def hazen_williams_headloss(length, roughness, diameter, flow):
    return ((10.67 * length) / ((roughness ** 1.852) * (diameter ** 4.8704))) * flow * (abs(flow) ** 0.852)    

def darcy_weisbach_headloss(length, roughness, diameter, flow):
    f = (2 * np.log10((3.7 * diameter) / roughness)) ** (-2)
    return ((8 * f * length) / ((np.pi ** 2) * 9.80665 * (diameter ** 5))) * flow * abs(flow)

class Pipe:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, length: float, diameter: float, roughness: float, check_valve: bool):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.incident_nodes = [start_node, end_node]
        self.length = length
        self.diameter = diameter
        self.roughness = roughness
        self.check_valve = check_valve

    def add_flow_bounds_from_heads(self, head_loss_equation):
        forward_head_diff = np.maximum(self.start_node.head_upper_bound - self.end_node.head_lower_bound, 0.0)
        backward_head_diff = np.maximum(self.end_node.head_upper_bound - self.start_node.head_lower_bound, 0.0)

        if head_loss_equation == 'HW':
            self.loss_coefficient = (10.67 * self.length) / ((self.roughness ** 1.852) * (self.diameter ** 4.8704))
            self.flow_upper_bound =  (forward_head_diff / self.loss_coefficient) ** (1 / 1.852)
            self.flow_lower_bound = -((backward_head_diff / self.loss_coefficient) ** (1 / 1.852)) if not self.check_valve else np.zeros(len(backward_head_diff))
        else: #Reynolds dependent roughness coeff? Currently, we use Colebrook approximation
            self.loss_coefficient = (8 * self.length * ((2 * np.log10((3.7 * self.diameter) / self.roughness)) ** (-2))) / ((np.pi ** 2) * 9.80665 * (self.diameter ** 5))
            self.flow_upper_bound =  ((forward_head_diff / self.loss_coefficient) ** 0.5)
            self.flow_lower_bound = -((backward_head_diff / self.loss_coefficient) ** 0.5) if not self.check_valve else np.zeros(len(backward_head_diff))

class Valve:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, diameter: float, valve_type: str):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.incident_nodes = [start_node, end_node]
        self.diameter = diameter
        self.valve_type = valve_type

    def add_flow_bounds(self, flow_lower_bound: np.array, flow_upper_bound: np.array):
        self.flow_lower_bound = flow_lower_bound
        self.flow_upper_bound = flow_upper_bound

class Pump:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, pump_type: str, fixed_cost: np.array, operational_cost: np.array):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.incident_nodes = [start_node, end_node]
        self.pump_type = pump_type
        self.fixed_cost = fixed_cost
        self.operational_cost = operational_cost

    def add_flow_bounds(self, flow_upper_bound: np.array):
        self.flow_lower_bound = np.zeros(len(flow_upper_bound))
        self.flow_upper_bound = flow_upper_bound

    def add_pump_costs(self, fixed_cost: np.array, operational_cost: np.array):
        self.fixed_cost = fixed_cost
        self.operational_cost = operational_cost

def fetch_epanet_data(water_network_path, head_loss_equation):
    water_network = wntr.network.WaterNetworkModel(water_network_path)

    patterns = {}
    curves = {}
    junctions = {}
    tanks = {}
    pumps = {}
    valves = {}
    pipes = {}

    duration = water_network.options.time.duration
    hydraulic_step_length = water_network.options.time.hydraulic_timestep
    pattern_step_length = water_network.options.time.pattern_timestep
    pattern_hydraulic_ratio = int(pattern_step_length / hydraulic_step_length)
    time_steps = int(duration / hydraulic_step_length)

    def expand_pattern(multipliers: np.array, base: float) -> np.array:
        cycle = np.repeat(multipliers * base, pattern_hydraulic_ratio)
        n_repeats = math.ceil(time_steps / cycle.size)
        return np.tile(cycle, n_repeats)[:time_steps]

    global_pump_efficiency = water_network.options.energy.global_efficiency
    global_energy_price = water_network.options.energy.global_price
    global_energy_price_pattern = water_network.options.energy.global_pattern
    global_demand_pattern = water_network.options.hydraulic.pattern

    for pattern_name, each_pattern in water_network.patterns():
        patterns[pattern_name] = each_pattern.multipliers

    for curve_name, each_curve in water_network.curves():
        curves[curve_name] = each_curve.points

    for reservoir_name, each_reservoir in water_network.reservoirs():
        reservoir_head = expand_pattern(patterns[each_reservoir.head_pattern_name], each_reservoir.base_head) if each_reservoir.head_pattern_name in patterns else np.repeat(each_reservoir.base_head, time_steps)

        junctions[reservoir_name] = Junction(reservoir_name, 'reservoir', reservoir_head, None)
        junctions[reservoir_name].add_head_bounds(reservoir_head, reservoir_head)

    for tank_name, each_tank in water_network.tanks():
        diameter = each_tank.diameter
        area = ((diameter/2) ** 2) * np.pi
        storage_lower_bound = each_tank.min_level * area
        storage_upper_bound = each_tank.max_level * area
        storage_initial = each_tank.init_level * area

        tanks[tank_name] = Tank(tank_name, each_tank.elevation, diameter, area, storage_initial, storage_lower_bound, storage_upper_bound)
        tanks[tank_name].add_head_bounds(np.repeat(each_tank.elevation + each_tank.min_level, time_steps), np.repeat(each_tank.elevation + each_tank.max_level, time_steps))

    global_head_upper_bound = np.max([node.head_upper_bound for node in (junctions | tanks).values()], axis=0)

    for junction_name, each_junction in water_network.junctions():
        junction_demand_pattern = each_junction.demand_pattern if each_junction.demand_pattern in patterns else global_demand_pattern
        net_demand = expand_pattern(patterns[junction_demand_pattern], each_junction.base_demand) if junction_demand_pattern is not None else np.repeat(each_junction.base_demand, time_steps)
        junction_elevation = np.repeat(each_junction.elevation, time_steps)

        junctions[junction_name] = Junction(junction_name, 'junction', junction_elevation, net_demand)
        junctions[junction_name].add_head_bounds(junction_elevation, global_head_upper_bound)

    nodes = junctions | tanks

    for pipe_name, each_pipe in water_network.pipes():
        length = each_pipe.length
        diameter = each_pipe.diameter
        roughness = each_pipe.roughness
        check_valve = each_pipe.check_valve

        start_node = nodes[each_pipe.start_node_name]
        end_node = nodes[each_pipe.end_node_name]

        pipes[pipe_name] = Pipe(pipe_name, start_node, end_node, length, diameter, roughness, check_valve)
        start_node.add_outgoing_edge(pipes[pipe_name])
        end_node.add_incoming_edge(pipes[pipe_name])
        pipes[pipe_name].add_flow_bounds_from_heads(head_loss_equation)
    
    global_flow_upper_bound = np.max([pipe.flow_upper_bound for pipe in pipes.values()], axis=0)
    global_flow_lower_bound = np.min([pipe.flow_lower_bound for pipe in pipes.values()], axis=0)

    for pump_name, each_pump in water_network.pumps():
        base_energy_price = each_pump.energy_price or global_energy_price or 1.0
        energy_price_pattern = each_pump.energy_pattern or global_energy_price_pattern
        energy_price = expand_pattern(patterns[energy_price_pattern], base_energy_price) if energy_price_pattern is not None else np.repeat(base_energy_price, time_steps)

        start_node = nodes[each_pump.start_node_name]
        end_node = nodes[each_pump.end_node_name]

        pumps[pump_name] = Pump(pump_name, start_node, end_node, each_pump.pump_type, np.repeat(100.0, time_steps), energy_price)
        start_node.add_outgoing_edge(pumps[pump_name])
        end_node.add_incoming_edge(pumps[pump_name])
        pumps[pump_name].add_flow_bounds(global_flow_upper_bound)

    for valve_name, each_valve in water_network.valves():
        diameter = each_valve.diameter
        valve_type = each_valve.valve_type

        start_node = nodes[each_valve.start_node_name]
        end_node = nodes[each_valve.end_node_name]

        valves[valve_name] = Valve(valve_name, start_node, end_node, diameter, valve_type)
        start_node.add_outgoing_edge(valves[valve_name])
        end_node.add_incoming_edge(valves[valve_name])
        valves[valve_name].add_flow_bounds(global_flow_lower_bound, global_flow_upper_bound)

    return junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_step_length

if __name__ == '__main__':
    head_loss_equation = 'HW'
    print_water_network_info('data/epanet/Net1.inp')
    #fetch_epanet_data('data/epanet/Net1.inp', head_loss_equation)
    #fetch_epanet_data('data/epanet/st-net3-3.inp', head_loss_equation)