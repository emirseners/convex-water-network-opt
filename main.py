import os
import time
from src.read_epanet import epanet_data
from src.soc_outer import define_soc_constraints
from src.models.pyomo import PyomoModel
from src.models.gurobi import GurobiModel
from src.valid_inequalities import fix_bridge_flows, node_subset
from src.helpers import write_results, write_bounds, write_valid_inequalities
from src.bound_tightening import mosek_obbt
from src.hull_facets import find_hydraulic_loss_facets, find_head_gain_facets
from src.algorithms import two_phased, two_phased_soc, callback_soc

if __name__ == '__main__':
    head_loss_equation = 'HW'
    flow_scale, head_scale = 16, 0.25

    water_networks = [
        ('Simple_FSD-24_Steps-Day_1', 'operational'),
        ('Simple_FSD-24_Steps-Day_2', 'operational'),
        ('ATM-12_Steps-Day_1', 'operational'),
        ('ATM-12_Steps-Day_2', 'operational'),
        ('Poormond-12_Steps-Day_1', 'operational'),
        ('Poormond-12_Steps-Day_2', 'operational')
    ]

    '''water_networks = [
        ('blacksburg', 'design'),
        ('foss_iron', 'design'),
        ('foss_poly_0', 'design'),
        ('foss_poly_1', 'design'),
        ('hanoi', 'design'),
        #('modena', 'design'),
        ('NewYork', 'design'),
        ('pescara', 'design'),
        ('shamir', 'design')
    ]'''

    methods = ['two_phased_gurobi_soc', 'gurobi_callback']
    obbt_flags = [True]
    valid_inequalities_flag = False
    cone_budget = 4
    run_prefix = f'exp_operational_{9}'
    time_limit, optimality_gap = 1200, 0.001
    binary_flow_direction, binary_pump_valve, binary_pipe_selection = True, True, False

    for obbt_flag in obbt_flags:
        run_name = f'{run_prefix}_obbt({obbt_flag})'
        results_folder = os.path.join('results', run_name)
        os.makedirs(results_folder, exist_ok=True)
        write_results(results_folder, 'results.txt', 'Model | Method | Status | LB | UB | Incumbent Count | Hydraulic Feasible Count | Solve.T. | Obbt T. | Hull Facet T.', mode='w')

        for epanet_model, problem in water_networks:
            network = epanet_data(f'data/{problem}/{epanet_model}', problem, head_loss_equation, flow_scale, head_scale)

            results_directory = os.path.join(results_folder, epanet_model)
            os.makedirs(results_directory, exist_ok=True)
            network.results_directory, network.time_limit, network.optimality_gap, network.problem = results_directory, time_limit, optimality_gap, problem

            fix_bridge_flows(network)

            if obbt_flag:
                write_bounds(network, sheet_name='pre_obbt', output_file='variable_bounds.xlsx')
                obbt_start = time.time()
                horizons = [(period, period + 1) for period in range(network.time_steps)]
                mosek_obbt(network, horizons, binary_flow_direction=False, binary_pump_valve=False, binary_pipe_selection=False)
                #horizons = [(0, network.time_steps)] * network.time_steps
                #mosek_obbt(network, horizons, binary_flow_direction=False, binary_pump_valve=False, binary_pipe_selection=False, mode='a')
                obbt_time = time.time() - obbt_start

            for _ in range(10):
                for link in (network.pipes | network.pumps | network.valves).values():
                    link.add_flow_bounds()
            write_bounds(network, sheet_name='final_bounds', output_file='variable_bounds.xlsx')

            valid_inequalities = set()

            # Will work on it later
            '''if valid_inequalities_flag:
                horizons = [[t, t + 1] for t in range(network.time_steps)]
                horizons.append([0, network.time_steps])
                valid_inequalities.update(node_subset(network.junctions | network.tanks, network.pipes | network.pumps | network.valves, horizons, max_cardinality=3))'''

            network.valid_inequalities = valid_inequalities
            write_valid_inequalities(valid_inequalities, results_directory)

            for method in methods:
                hull_facet_time, incumbent_count, hydraulic_feasible_count = None, None, None
                if method in ('two_phased_mosek', 'two_phased_gurobi_soc', 'gurobi_callback'):
                    hull_facet_start = time.time()
                    find_hydraulic_loss_facets(network.pipes | network.valves, network.time_steps)
                    find_head_gain_facets(network.pumps, network.time_steps)
                    hull_facet_time = time.time() - hull_facet_start

                if method == 'gurobi':
                    status, solution, lower_bound, upper_bound, solution_time = GurobiModel('gurobi', network)
                elif method == 'baron':
                    status, solution, lower_bound, upper_bound, solution_time = PyomoModel(method, network)
                elif method == 'two_phased_mosek':
                    status, solution, lower_bound, upper_bound, solution_time, incumbent_count, hydraulic_feasible_count = two_phased(network, binary_flow_direction, binary_pump_valve, binary_pipe_selection)
                elif method == 'two_phased_gurobi_soc':
                    define_soc_constraints(network, cone_budget)
                    status, solution, lower_bound, upper_bound, solution_time, incumbent_count, hydraulic_feasible_count = two_phased_soc(network, binary_flow_direction, binary_pump_valve, binary_pipe_selection)
                elif method == 'gurobi_callback':
                    define_soc_constraints(network, cone_budget)
                    status, solution, lower_bound, upper_bound, solution_time, incumbent_count, hydraulic_feasible_count = callback_soc(network, binary_flow_direction, binary_pump_valve, binary_pipe_selection)

                write_results(results_folder, 'results.txt', f'{epanet_model} | {method} | {status} | {f"{lower_bound:.2f}" if lower_bound is not None else "-"} | {f"{upper_bound:.2f}" if upper_bound is not None else "-"} | {f"{incumbent_count}" if incumbent_count is not None else "-"} | {f"{hydraulic_feasible_count}" if hydraulic_feasible_count is not None else "-"} | {solution_time:.2f} | {f"{obbt_time:.2f}" if obbt_flag else "-"} | {f"{hull_facet_time:.2f}" if hull_facet_time is not None else "-"}')