from __future__ import annotations
from .helpers import round_bound
import numpy as np
import wntr

def print_water_network_info(water_network_path):
    water_network = wntr.network.WaterNetworkModel(water_network_path + '.inp')

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

def get_head_bounds(node):
    if isinstance(node, Tank):
        return node.head_lower_bound[:-1], node.head_upper_bound[:-1]
    return node.head_lower_bound, node.head_upper_bound

def get_head_difference_bounds(edge):
    start_lb, start_ub = get_head_bounds(edge.start_node)
    end_lb, end_ub = get_head_bounds(edge.end_node)
    return start_lb - end_ub, start_ub - end_lb

def head_loss(edge, flow):
    return (edge.resistance_coefficient * np.sign(flow) * np.abs(flow) ** edge.exponent)

class Network:
    def __init__(self, junctions, tanks, pipes, pumps, valves, time_steps, pipe_families):
        self.junctions, self.tanks, self.pipes, self.pumps, self.valves, self.time_steps, self.pipe_families = junctions, tanks, pipes, pumps, valves, time_steps, pipe_families

class Junction:
    def __init__(self, label: str, segment: str, elevation: np.array, net_demand: np.array, time_steps: int):
        self.label = label
        self.segment = segment
        self.net_demand = net_demand
        self.elevation = elevation
        self.incoming_edges = []
        self.outgoing_edges = []
        self.head_lower_bound = np.full(time_steps, -np.inf)
        self.head_upper_bound = np.full(time_steps, np.inf)

    def add_incoming_edge(self, incoming_edge: Pipe | Valve | Pump):
        self.incoming_edges.append(incoming_edge)

    def add_outgoing_edge(self, outgoing_edge: Pipe | Valve | Pump):
        self.outgoing_edges.append(outgoing_edge)

    def add_head_bounds(self, head_lower_bound=None, head_upper_bound=None):
        if head_lower_bound is not None:
            self.head_lower_bound = np.maximum(head_lower_bound, self.head_lower_bound)
        if head_upper_bound is not None:
            self.head_upper_bound = np.minimum(head_upper_bound, self.head_upper_bound)

class Tank:
    def __init__(self, label: str, area: float, tank_coef: float, time_steps: int):
        self.label = label
        self.area = area
        self.tank_coef = tank_coef
        self.incoming_edges = []
        self.outgoing_edges = []
        self.head_lower_bound = np.full(time_steps + 1, -np.inf)
        self.head_upper_bound = np.full(time_steps + 1, np.inf)

    def add_incoming_edge(self, incoming_edge: Pipe | Valve | Pump):
        self.incoming_edges.append(incoming_edge)

    def add_outgoing_edge(self, outgoing_edge: Pipe | Valve | Pump):
        self.outgoing_edges.append(outgoing_edge)

    def add_head_bounds(self, head_lower_bound=None, head_upper_bound=None):
        if head_lower_bound is not None:
            if head_lower_bound.shape == self.head_lower_bound[:-1].shape:
                self.head_lower_bound[:-1] = np.maximum(head_lower_bound, self.head_lower_bound[:-1])
            else:
                self.head_lower_bound = np.maximum(head_lower_bound, self.head_lower_bound)
        if head_upper_bound is not None:
            if head_upper_bound.shape == self.head_upper_bound[:-1].shape:
                self.head_upper_bound[:-1] = np.minimum(head_upper_bound, self.head_upper_bound[:-1])
            else:
                self.head_upper_bound = np.minimum(head_upper_bound, self.head_upper_bound)

class PipeFamily:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, options: np.array, exponent: float, is_optional = False):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.options = options
        self.exponent = exponent
        self.is_optional = is_optional

    def declare_family(self):
        for pipe in self.options:
            pipe.pipe_family = self

class Pipe:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, resistance_coefficient: float, exponent: float, time_steps: int, installation_cost = None, selection_decision = 1):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.exponent = exponent
        self.resistance_coefficient = resistance_coefficient
        self.installation_cost = installation_cost
        self.flow_lower_bound = np.full(time_steps, -np.inf)
        self.flow_upper_bound = np.full(time_steps, np.inf)
        self.positive_flow_lower_bound = np.zeros(time_steps)
        self.negative_flow_upper_bound = np.zeros(time_steps)
        self.direction_decision = np.full(time_steps, -1, dtype=int)
        self.selection_decision = selection_decision
        self.fixed_flow = np.full(time_steps, False, dtype=bool)
        self.pipe_family = None

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.minimum(flow_upper_bound, self.flow_upper_bound)
        self.add_directional_flow_bounds(self.flow_lower_bound, self.flow_upper_bound)

    def add_directional_flow_bounds(self, positive_flow_lower_bound=None, negative_flow_upper_bound=None):
        if positive_flow_lower_bound is not None:
            self.positive_flow_lower_bound = np.maximum(positive_flow_lower_bound, self.positive_flow_lower_bound)
        if negative_flow_upper_bound is not None:
            self.negative_flow_upper_bound = np.minimum(negative_flow_upper_bound, self.negative_flow_upper_bound)
        self.propagate_bounds()

    def propagate_bounds(self, epsilon=1e-9):
        if self.selection_decision != 0:
            head_diff_lb, head_diff_ub = get_head_difference_bounds(self)
            positive_head_diff_lb = np.maximum(head_diff_lb, head_loss(self, self.positive_flow_lower_bound))
            positive_head_diff_ub = np.minimum(head_diff_ub, head_loss(self, self.flow_upper_bound))
            negative_head_diff_lb = np.maximum(head_diff_lb, head_loss(self, self.flow_lower_bound))
            negative_head_diff_ub = np.minimum(head_diff_ub, head_loss(self, self.negative_flow_upper_bound))
            positive_possible = ((self.direction_decision != 0) & (self.positive_flow_lower_bound <= self.flow_upper_bound) & (positive_head_diff_lb <= positive_head_diff_ub + epsilon))
            negative_possible = ((self.direction_decision != 1) & (self.flow_lower_bound <= self.negative_flow_upper_bound) & (negative_head_diff_lb <= negative_head_diff_ub + epsilon))

            if self.selection_decision == -1:
                if np.any((self.flow_lower_bound > 0.0) | (self.flow_upper_bound < 0.0)):
                    self.selection_decision = 1
                elif not np.all(positive_possible | negative_possible):
                    self.selection_decision = 0

            if self.selection_decision != 0:
                positive_flow_lb = np.maximum(self.positive_flow_lower_bound, round_bound((np.maximum(positive_head_diff_lb, 0.0) / self.resistance_coefficient) ** (1 / self.exponent), 'floor'))
                positive_flow_ub = np.minimum(self.flow_upper_bound, round_bound((np.maximum(positive_head_diff_ub, 0.0) / self.resistance_coefficient) ** (1 / self.exponent), 'ceiling'))
                negative_flow_lb = np.maximum(self.flow_lower_bound, round_bound(-((np.maximum(-negative_head_diff_lb, 0.0) / self.resistance_coefficient) ** (1 / self.exponent)), 'floor'))
                negative_flow_ub = np.minimum(self.negative_flow_upper_bound, round_bound(-((np.maximum(-negative_head_diff_ub, 0.0) / self.resistance_coefficient) ** (1 / self.exponent)), 'ceiling'))

                self.positive_flow_lower_bound = np.where(positive_possible, positive_flow_lb, self.positive_flow_lower_bound)
                self.negative_flow_upper_bound = np.where(negative_possible, negative_flow_ub, self.negative_flow_upper_bound)
                flow_lb = np.where(negative_possible, negative_flow_lb, positive_flow_lb)
                flow_ub = np.where(positive_possible, positive_flow_ub, negative_flow_ub)
                self.flow_lower_bound = np.maximum(self.flow_lower_bound, np.minimum(flow_lb, 0.0) if self.selection_decision == -1 else flow_lb)
                self.flow_upper_bound = np.minimum(self.flow_upper_bound, np.maximum(flow_ub, 0.0) if self.selection_decision == -1 else flow_ub)

                if self.selection_decision == 1:
                    head_diff_lb = np.where(negative_possible, negative_head_diff_lb, positive_head_diff_lb)
                    head_diff_ub = np.where(positive_possible, positive_head_diff_ub, negative_head_diff_ub)
                    start_lb, start_ub = get_head_bounds(self.start_node)
                    end_lb, end_ub = get_head_bounds(self.end_node)
                    self.start_node.add_head_bounds(round_bound(end_lb + head_diff_lb, 'floor'), round_bound(end_ub + head_diff_ub, 'ceiling'))
                    self.end_node.add_head_bounds(round_bound(start_lb - head_diff_ub, 'floor'), round_bound(start_ub - head_diff_lb, 'ceiling'))

        if self.selection_decision == 0:
            self.flow_lower_bound = np.maximum(self.flow_lower_bound, 0.0)
            self.flow_upper_bound = np.minimum(self.flow_upper_bound, 0.0)

        if self.selection_decision != 0:
            unknown = self.direction_decision == -1
            choose_positive = (unknown & positive_possible & (self.flow_lower_bound >= 0.0))
            choose_negative = (unknown & negative_possible & (self.flow_upper_bound <= 0.0) & ~choose_positive)
            self.direction_decision[choose_positive] = 1
            self.direction_decision[choose_negative] = 0

        self.fixed_flow = self.flow_lower_bound == self.flow_upper_bound

        if self.selection_decision == 1 and self.pipe_family is not None:
            for pipe in self.pipe_family.options:
                if pipe is self or pipe.selection_decision == 0: continue
                pipe.selection_decision = 0
                pipe.add_flow_bounds(0.0, 0.0)

        elif self.pipe_family is not None:
            remaining = [pipe for pipe in self.pipe_family.options if pipe.selection_decision != 0]
            if not self.pipe_family.is_optional and len(remaining) == 1 and remaining[0].selection_decision == -1:
                remaining[0].selection_decision = 1
                remaining[0].add_flow_bounds()

class Valve:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, resistance_coefficient: float, exponent: float, time_steps: int):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.exponent = exponent
        self.resistance_coefficient = resistance_coefficient
        self.flow_lower_bound = np.zeros(time_steps)
        self.flow_upper_bound = np.full(time_steps, np.inf)
        self.positive_flow_lower_bound = np.zeros(time_steps)
        self.status_decisions = np.full(time_steps, -1, dtype=int)
        self.fixed_flow = np.full(time_steps, False, dtype=bool)

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.minimum(flow_upper_bound, self.flow_upper_bound)
        self.add_directional_flow_bounds(self.flow_lower_bound)

    def add_directional_flow_bounds(self, positive_flow_lower_bound=None):
        if positive_flow_lower_bound is not None:
            self.positive_flow_lower_bound = np.maximum(positive_flow_lower_bound, self.positive_flow_lower_bound)
        self.propagate_bounds()

    def propagate_bounds(self, epsilon=1e-9):
        head_diff_lb, head_diff_ub = get_head_difference_bounds(self)
        active_head_lb = np.maximum(head_diff_lb, head_loss(self, self.positive_flow_lower_bound))
        active_head_ub = np.minimum(head_diff_ub, head_loss(self, self.flow_upper_bound))
        active_possible = active_head_lb <= active_head_ub + epsilon
        unknown = self.status_decisions == -1
        self.status_decisions[unknown & (self.flow_lower_bound > 0.0)] = 1
        self.status_decisions[unknown & ~active_possible] = 0

        flow_lb = np.maximum(self.positive_flow_lower_bound, round_bound((np.maximum(active_head_lb, 0.0) / self.resistance_coefficient) ** (1 / self.exponent), 'floor'))
        flow_ub = np.minimum(self.flow_upper_bound, round_bound((np.maximum(active_head_ub, 0.0) / self.resistance_coefficient) ** (1 / self.exponent), 'ceiling'))
        self.positive_flow_lower_bound = np.where(active_possible, flow_lb, self.positive_flow_lower_bound)
        self.flow_lower_bound = np.maximum(self.flow_lower_bound, np.where(self.status_decisions == 1, self.positive_flow_lower_bound, 0.0))
        self.flow_upper_bound = np.minimum(self.flow_upper_bound, np.where(self.status_decisions == 0, 0.0, flow_ub))
        self.fixed_flow = self.flow_lower_bound == self.flow_upper_bound

        head_diff_lb = np.where(self.status_decisions == 1, active_head_lb, -np.inf)
        head_diff_ub = np.where(self.status_decisions == 1, active_head_ub, np.inf)
        start_lb, start_ub = get_head_bounds(self.start_node)
        end_lb, end_ub = get_head_bounds(self.end_node)
        self.start_node.add_head_bounds(round_bound(end_lb + head_diff_lb, 'floor'), round_bound(end_ub + head_diff_ub, 'ceiling'))
        self.end_node.add_head_bounds(round_bound(start_lb - head_diff_ub, 'floor'), round_bound(start_ub - head_diff_lb, 'ceiling'))

class Pump:
    def __init__(self, label: str, start_node: Junction | Tank, end_node: Junction | Tank, time_steps: int, fixed_cost: np.array, operational_cost: np.array):
        self.label = label
        self.start_node = start_node
        self.end_node = end_node
        self.fixed_cost = fixed_cost
        self.operational_cost = operational_cost
        self.flow_lower_bound = np.zeros(time_steps)
        self.flow_upper_bound = np.full(time_steps, np.inf)
        self.positive_flow_lower_bound = np.zeros(time_steps)
        self.status_decisions = np.full(time_steps, -1, dtype=int)
        self.fixed_flow = np.full(time_steps, False, dtype=bool)

    def add_flow_bounds(self, flow_lower_bound=None, flow_upper_bound=None):
        if flow_lower_bound is not None:
            self.flow_lower_bound = np.maximum(flow_lower_bound, self.flow_lower_bound)
        if flow_upper_bound is not None:
            self.flow_upper_bound = np.minimum(flow_upper_bound, self.flow_upper_bound)
        self.add_directional_flow_bounds(self.flow_lower_bound)

    def add_directional_flow_bounds(self, positive_flow_lower_bound=None):
        if positive_flow_lower_bound is not None:
            self.positive_flow_lower_bound = np.maximum(positive_flow_lower_bound, self.positive_flow_lower_bound)
        self.propagate_bounds()

    def propagate_bounds(self, epsilon=1e-9):
        head_diff_lb, head_diff_ub = get_head_difference_bounds(self)
        active_head_lb = np.maximum(head_diff_lb, -self.curve_a - self.curve_b * self.positive_flow_lower_bound ** self.curve_c)
        active_head_ub = np.minimum(head_diff_ub, -self.curve_a - self.curve_b * self.flow_upper_bound ** self.curve_c)
        active_possible = active_head_lb <= active_head_ub + epsilon
        unknown = self.status_decisions == -1
        self.status_decisions[unknown & (self.flow_lower_bound > 0.0)] = 1
        self.status_decisions[unknown & ~active_possible] = 0

        flow_lb = np.maximum(self.positive_flow_lower_bound, round_bound((np.maximum(active_head_lb + self.curve_a, 0.0) / -self.curve_b) ** (1 / self.curve_c), 'floor'))
        flow_ub = np.minimum(self.flow_upper_bound, round_bound((np.maximum(active_head_ub + self.curve_a, 0.0) / -self.curve_b) ** (1 / self.curve_c), 'ceiling'))
        self.positive_flow_lower_bound = np.where(active_possible, flow_lb, self.positive_flow_lower_bound)
        self.flow_lower_bound = np.maximum(self.flow_lower_bound, np.where(self.status_decisions == 1, self.positive_flow_lower_bound, 0.0))
        self.flow_upper_bound = np.minimum(self.flow_upper_bound, np.where(self.status_decisions == 0, 0.0, flow_ub))
        self.fixed_flow = self.flow_lower_bound == self.flow_upper_bound

        head_diff_lb = np.where(self.status_decisions == 1, active_head_lb, -np.inf)
        head_diff_ub = np.where(self.status_decisions == 1, active_head_ub, np.inf)
        start_lb, start_ub = get_head_bounds(self.start_node)
        end_lb, end_ub = get_head_bounds(self.end_node)
        self.start_node.add_head_bounds(round_bound(end_lb + head_diff_lb, 'floor'), round_bound(end_ub + head_diff_ub, 'ceiling'))
        self.end_node.add_head_bounds(round_bound(start_lb - head_diff_ub, 'floor'), round_bound(start_ub - head_diff_lb, 'ceiling'))

    def add_pump_costs(self, fixed_cost: np.array, operational_cost: np.array):
        self.fixed_cost = fixed_cost
        self.operational_cost = operational_cost

    #We only consider HEAD pumps. POWER pumps and variable speed pumps are neglected
    def add_head_curve(self, curve_a, curve_b, curve_c):
        self.curve_a, self.curve_b, self.curve_c = curve_a, curve_b, curve_c

def design_data(water_network_path, head_loss_equation, flow_scale, head_scale):
    with open(water_network_path + ".dat", encoding="utf-8-sig") as data_file:
        lines = [line.split("#", 1)[0].strip() for line in data_file]

    text = "\n".join(line for line in lines if line)
    tables = {}

    for statement in text.split(";"):
        if not statement.strip().startswith("param:"):
            continue

        header, body = statement.split(":=", 1)
        _, name, column_text = header.split(":", 2)
        columns = column_text.split()
        rows = {}

        for line in body.splitlines():
            tokens = line.split()
            if tokens:
                rows[tokens[0]] = {column: (value if column in ["node1", "node2", "pipe"] else float(value)) for column, value in zip(columns, tokens[1:])}

        tables[name.strip().lower()] = rows

    exponent = 1.852 if head_loss_equation == 'HW' else 2
    fixed_pipe_diameters = {row["pipe"]: row["fixdiam"] for row in tables.get("fixeddiam", {}).values()}
    junctions = {}
    pipes = {}
    pipe_families = {}

    for junction_name, row in tables["node"].items():
        is_reservoir = row["demand"] == 0.0 and row["pmin"] == row["pmax"]
        junctions[junction_name] = Junction(junction_name, "reservoir" if is_reservoir else "junction", np.array([head_scale * row["elev"]]), None if is_reservoir else np.array([flow_scale * row["demand"]]), 1)
        junctions[junction_name].add_head_bounds(np.array([head_scale * (row["elev"] + row["pmin"])]), np.array([head_scale * (row["elev"] + row["pmax"])]))

    for pipe_name, row in tables['pipe'].items():
        start_node = junctions[row['node1']]
        end_node = junctions[row['node2']]
        existing_diameter = row.get('exdiam', 0.0)
        family_members = []

        for resistance_label, diameter_dict in tables['diamset'].items():
            diameter = diameter_dict['discdiam']
            if pipe_name in fixed_pipe_diameters and diameter != fixed_pipe_diameters[pipe_name]: continue

            if head_loss_equation == 'HW':
                resistance_coefficient = (head_scale * ((10.66682948893005 * row['length']) / ((row['dKc'] ** 1.852) * (diameter ** 4.871)))) / (flow_scale ** 1.852)
            else:
                resistance_coefficient = (head_scale * ((8 * row['length'] * ((2 * np.log10((3.7 * diameter) / row['dKc'])) ** (-2))) / ((np.pi ** 2) * 9.80665 * (diameter ** 5)))) / (flow_scale ** 2)

            candidate_label = f'{pipe_name}_{resistance_label}'
            selection_decision = -1
            if existing_diameter == 0.0 and (pipe_name in fixed_pipe_diameters or len(tables['diamset']) == 1):
                selection_decision = 1

            pipes[candidate_label] = Pipe(candidate_label, start_node, end_node, resistance_coefficient, exponent, 1, row['length'] * diameter_dict['disccost'], selection_decision=selection_decision)

            flow_capacity = (flow_scale * row['vmax'] * np.pi * (diameter ** 2)) / 4
            pipes[candidate_label].add_flow_bounds([-flow_capacity], [flow_capacity])
            start_node.add_outgoing_edge(pipes[candidate_label])
            end_node.add_incoming_edge(pipes[candidate_label])
            family_members.append(pipes[candidate_label])

        pipe_families[pipe_name] = PipeFamily(pipe_name, start_node, end_node, np.array(family_members, dtype=object), family_members[0].exponent, existing_diameter > 0.0)

        if existing_diameter > 0.0:
            if head_loss_equation == 'HW':
                resistance_coefficient = (head_scale * ((10.66682948893005 * row['length']) / ((row['dKc'] ** 1.852) * (existing_diameter ** 4.871)))) / (flow_scale ** 1.852)
            else:
                resistance_coefficient = (head_scale * ((8 * row['length'] * ((2 * np.log10((3.7 * existing_diameter) / row['dKc'])) ** (-2))) / ((np.pi ** 2) * 9.80665 * (existing_diameter ** 5)))) / (flow_scale ** 2)

            existing_label = f'{pipe_name}_EX'
            pipes[existing_label] = Pipe(existing_label, start_node, end_node, resistance_coefficient, exponent, 1)

            flow_capacity = (flow_scale * row['vmax'] * np.pi * (existing_diameter ** 2)) / 4
            pipes[existing_label].add_flow_bounds([-flow_capacity], [flow_capacity])
            start_node.add_outgoing_edge(pipes[existing_label])
            end_node.add_incoming_edge(pipes[existing_label])
            pipe_families[existing_label] = PipeFamily(existing_label, start_node, end_node, np.array([pipes[existing_label]], dtype=object), pipes[existing_label].exponent)

    for pipe_family in pipe_families.values():
        pipe_family.declare_family()

    for _ in range(10):
        for edge in pipes.values():
            edge.add_flow_bounds()

    network = Network(junctions, {}, pipes, {}, {}, 1, pipe_families)
    return network

def epanet_data(water_network_path, problem, head_loss_equation, flow_scale, head_scale):
    if problem == 'design':
        return design_data(water_network_path, head_loss_equation, flow_scale, head_scale)

    water_network = wntr.network.WaterNetworkModel(water_network_path + '.inp')

    patterns = {}
    junctions = {}
    tanks = {}
    pumps = {}
    valves = {}
    pipes = {}

    hydraulic_timestep = int(water_network.options.time.hydraulic_timestep)
    pattern_step_length = int(water_network.options.time.pattern_timestep)
    pattern_start = int(water_network.options.time.pattern_start)
    time_steps = int(int(water_network.options.time.duration) / hydraulic_timestep)

    def expand_pattern(multipliers: np.array, base: float) -> np.array:
        multipliers = np.asarray(multipliers, dtype=float)

        if multipliers.size == 0:
            return np.full(time_steps, base, dtype=float)

        sample_times = (pattern_start + np.arange(time_steps) * hydraulic_timestep)
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
        junctions[reservoir_name] = Junction(reservoir_name, 'reservoir', head_scale * reservoir_elevation, None, time_steps)
        junctions[reservoir_name].add_head_bounds(head_scale * reservoir_elevation, head_scale * reservoir_elevation)

    for tank_name, each_tank in water_network.tanks():
        area = (flow_scale / head_scale) * round(((each_tank.diameter / 2) ** 2) * np.pi, 4)
        head_lower_bound = each_tank.elevation + np.insert(np.repeat(each_tank.min_level, time_steps), 0, each_tank.init_level)
        head_upper_bound = each_tank.elevation + np.insert(np.repeat(each_tank.max_level, time_steps), 0, each_tank.init_level)
        head_lower_bound[-1] = max(head_lower_bound[-1], each_tank.elevation + each_tank.init_level)
        tanks[tank_name] = Tank(tank_name, area, hydraulic_timestep / area, time_steps)
        tanks[tank_name].add_head_bounds(head_scale * head_lower_bound, head_scale * head_upper_bound)

    for junction_name, each_junction in water_network.junctions():
        junction_demand_pattern = each_junction.demand_pattern if each_junction.demand_pattern in patterns else global_demand_pattern
        net_demand = expand_pattern(patterns[junction_demand_pattern], each_junction.base_demand) if junction_demand_pattern is not None else np.repeat(each_junction.base_demand, time_steps)
        junction_elevation = head_scale * np.repeat(each_junction.elevation, time_steps)
        junctions[junction_name] = Junction(junction_name, 'junction', junction_elevation, net_demand * global_demand_multiplier * flow_scale, time_steps)

    nodes = junctions | tanks

    for pump_name, each_pump in water_network.pumps():
        base_energy_price = each_pump.energy_price or global_energy_price or 0.001
        energy_price_pattern = each_pump.energy_pattern or global_energy_price_pattern
        energy_price = expand_pattern(patterns[energy_price_pattern], base_energy_price) if energy_price_pattern is not None else np.repeat(base_energy_price, time_steps)
        price_coefficient = hydraulic_timestep * energy_price

        start_node = nodes[each_pump.start_node_name]
        end_node = nodes[each_pump.end_node_name]
        pumps[pump_name] = Pump(pump_name, start_node, end_node, time_steps, 54 * price_coefficient, (196 * price_coefficient) / flow_scale)

        start_node.add_outgoing_edge(pumps[pump_name])
        end_node.add_incoming_edge(pumps[pump_name])

        curve_a, curve_b, curve_c = each_pump.get_head_curve_coefficients()
        curve_c = curve_c if abs(curve_c - round(curve_c)) > 1e-4 else round(curve_c)
        pumps[pump_name].add_head_curve(curve_a * head_scale, -((curve_b * head_scale) / (flow_scale ** curve_c)), curve_c)

    exponent = 1.852 if head_loss_equation == 'HW' else 2
    for pipe_name, each_pipe in water_network.pipes():
        start_node = nodes[each_pipe.start_node_name]
        end_node = nodes[each_pipe.end_node_name]

        # Currently, all input files have HW. If DW, then one should adjust coefficients with adequate units
        if head_loss_equation == 'HW':
            resistance_coefficient = (head_scale * ((10.66682948893005 * each_pipe.length) / ((each_pipe.roughness ** 1.852) * (each_pipe.diameter ** 4.871)))) / (flow_scale ** 1.852)
        else: # Reynolds dependent roughness coeff? Currently, we use Colebrook approximation
            resistance_coefficient = (head_scale * ((8 * each_pipe.length * ((2 * np.log10((3.7 * each_pipe.diameter) / each_pipe.roughness)) ** (-2))) / ((np.pi ** 2) * 9.80665 * (each_pipe.diameter ** 5))))  / (flow_scale ** 2)

        if not each_pipe.check_valve:
            pipes[pipe_name] = Pipe(pipe_name, start_node, end_node, resistance_coefficient, exponent, time_steps)
            start_node.add_outgoing_edge(pipes[pipe_name])
            end_node.add_incoming_edge(pipes[pipe_name])
        else: # Currently, no instance exists with valves other than check valves, thus we only consider them
            valves[pipe_name] = Valve(pipe_name, start_node, end_node, resistance_coefficient, exponent, time_steps)
            start_node.add_outgoing_edge(valves[pipe_name])
            end_node.add_incoming_edge(valves[pipe_name])

    set_operational_bounds(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_scale)

    pipe_families = {}
    for pipe_name, pipe in pipes.items():
        pipe_families[pipe_name] = PipeFamily(pipe_name, pipe.start_node, pipe.end_node, np.array([pipe], dtype=object), pipe.exponent)

    for pipe_family in pipe_families.values():
        pipe_family.declare_family()

    network = Network(junctions, tanks, pipes, pumps, valves, time_steps, pipe_families)
    return network

def set_operational_bounds(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, head_scale, pressure_margin=100):
    for pump in pumps.values():
        flow_upper_bound = (-pump.curve_a / pump.curve_b) ** (1 / pump.curve_c)
        pump.add_flow_bounds(None, np.full(time_steps, flow_upper_bound))

    '''flow_capacity = np.array([np.maximum(junction.net_demand, 0) for junction in junctions.values() if junction.net_demand is not None]).sum(axis=0) + (np.sum([tank.area * (tank.head_upper_bound[1:] - tank.head_lower_bound[1:]) for tank in tanks.values()], axis=0) / hydraulic_timestep)
    for link in (pipes | pumps | valves).values():
        link.add_flow_bounds(-flow_capacity, flow_capacity)'''

    global_head_ub = np.max([junction.elevation for junction in junctions.values()] + [tank.head_upper_bound[:-1] for tank in tanks.values()], axis=0) + sum((pump.curve_a for pump in pumps.values()), 0.0)

    for junction in junctions.values():
        if junction.segment == 'reservoir': continue
        #junction.add_head_bounds(junction.elevation if min(junction.net_demand) > 0 else junction.elevation - (head_scale * pressure_margin), global_head_ub)
        junction.add_head_bounds(np.maximum(junction.elevation - (head_scale * pressure_margin), 0.0), global_head_ub + (head_scale * pressure_margin))

    for _ in range(10):
        for edge in (pipes | pumps | valves).values():
            edge.add_flow_bounds()

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
        pipe.positive_flow_lower_bound = pipe.positive_flow_lower_bound[time_steps]
        pipe.negative_flow_upper_bound = pipe.negative_flow_upper_bound[time_steps]
        pipe.direction_decision = pipe.direction_decision[time_steps]
        pipe.fixed_flow = pipe.flow_lower_bound == pipe.flow_upper_bound

    for valve in valves.values():
        valve.flow_lower_bound = valve.flow_lower_bound[time_steps]
        valve.flow_upper_bound = valve.flow_upper_bound[time_steps]
        valve.positive_flow_lower_bound = valve.positive_flow_lower_bound[time_steps]
        valve.status_decisions = valve.status_decisions[time_steps]
        valve.fixed_flow = valve.flow_lower_bound == valve.flow_upper_bound

    for pump in pumps.values():
        pump.fixed_cost = pump.fixed_cost[time_steps]
        pump.operational_cost = pump.operational_cost[time_steps]
        pump.flow_lower_bound = pump.flow_lower_bound[time_steps]
        pump.flow_upper_bound = pump.flow_upper_bound[time_steps]
        pump.positive_flow_lower_bound = pump.positive_flow_lower_bound[time_steps]
        pump.status_decisions = pump.status_decisions[time_steps]
        pump.fixed_flow = pump.flow_lower_bound == pump.flow_upper_bound

    return junctions, tanks, pipes, pumps, valves
