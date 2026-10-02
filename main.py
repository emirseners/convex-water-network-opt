import os
import time
from src.soc_outer import define_soc_constraints
from src.operational.pyomo import PyomoOperationalModel
from src.operational.gurobi import GurobiOperationalModel
from src.valid_inequalities import fix_bridge_flows, node_subset
from src.helpers import write_results, write_valid_inequalities
from src.bound_tightening import mosek_obbt, mosek_aggregate_inequalities
from src.hull_facets import find_hydraulic_loss_facets, find_head_gain_facets
from src.read_epanet import fetch_epanet_data, set_bounds, set_timesteps, preprocess_decisions
from src.algorithms import operational_problem_two_phased, operational_problem_two_phased_soc, operational_problem_soc_callback

if __name__ == '__main__':
    head_loss_equation = 'HW'
    flow_scale, head_scale = 16, 0.25

    water_networks = [
        ('Simple_FSD-24_Steps-Day_1', 'tasseff'),
        ('Simple_FSD-24_Steps-Day_2', 'tasseff'),
        ('ATM-12_Steps-Day_1', 'tasseff'),
        ('ATM-12_Steps-Day_2', 'tasseff'),
        ('Poormond-12_Steps-Day_1', 'tasseff'),
        ('Poormond-12_Steps-Day_2', 'tasseff')
    ]

    methods = ['two_phased_gurobi_soc', 'gurobi_callback']
    obbt_flags = [True]
    valid_inequalities_flag = False
    cone_budget = 4
    run_prefix = f'exp_{4}'
    time_limit = 1200

    for obbt_flag in obbt_flags:
        run_name = f'{run_prefix}_obbt({obbt_flag})'
        results_folder = os.path.join('results', run_name)
        os.makedirs(results_folder, exist_ok=True)
        write_results(results_folder, 'results.txt', 'Model | Method | Status | LB | UB | Incumbent Count | Hydraulic Feasible Count | Solve.T. | Obbt T. | Hull Facet T.', mode='w')

        for epanet_model, model_folder in water_networks:
            junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep = fetch_epanet_data(f'data/{model_folder}/{epanet_model}.inp', head_loss_equation, flow_scale, head_scale)

            results_directory = os.path.join(results_folder, epanet_model)
            os.makedirs(results_directory, exist_ok=True)

            set_bounds(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, results_directory, head_scale)
            fix_bridge_flows(junctions | tanks, pipes | pumps | valves)
            #time_steps = min(3, time_steps)
            #set_timesteps(junctions, tanks, pipes, pumps, valves, list(range(time_steps)))

            if obbt_flag:
                obbt_start = time.time()
                mosek_obbt(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, results_directory, [(period, period + 1) for period in range(time_steps)], time_limit, binary_pipe=False, binary_pump_valve=True, sheet_prefix="obbt_single", log_mode="w")
                #mosek_obbt(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, results_directory, [(0, time_steps)] * time_steps, time_limit, binary_pipe=False, binary_pump_valve=False, sheet_prefix="obbt_full", log_mode="a")
                obbt_time = time.time() - obbt_start
                preprocess_decisions(junctions, tanks, pipes, pumps, valves, time_steps, True, results_directory)

            valid_inequalities = set()

            if valid_inequalities_flag:
                variable_subsets = [[('pump_status', 'pumps'), ('flow_pipe_status', 'pipes')], [('pump_status', 'pumps')], [('flow_pipe_status', 'pipes')]]
                horizons = {(t, t + 1): None for t in range(time_steps)}
                horizons[(0, time_steps)] = None
                valid_inequalities.update(mosek_aggregate_inequalities(junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, results_directory, variable_subsets, horizons, time_limit, binary_pipe = False, binary_pump_valve = False, num_workers = 4))
                #valid_inequalities.update(node_subset(junctions | tanks, pipes | pumps | valves, hydraulic_timestep, horizons, max_cardinality=3))

            write_valid_inequalities(valid_inequalities, results_directory)

            for method in methods:
                hull_facet_time, incumbent_count, hydraulic_feasible_count = None, None, None
                if method in ('two_phased_mosek', 'two_phased_gurobi_soc', 'gurobi_callback'):
                    hull_facet_start = time.time()
                    find_hydraulic_loss_facets(pipes | valves, time_steps)
                    find_head_gain_facets(pumps, time_steps)
                    hull_facet_time = time.time() - hull_facet_start

                if method == 'gurobi':
                    status, solution, lower_bound, upper_bound, solution_time = GurobiOperationalModel('gurobi', results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit)
                elif method == 'two_phased_mosek':
                    status, solution, lower_bound, upper_bound, solution_time, incumbent_count, hydraulic_feasible_count = operational_problem_two_phased(results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit)
                elif method == 'two_phased_gurobi_soc':
                    define_soc_constraints(pipes, pumps, valves, time_steps, cone_budget)
                    status, solution, lower_bound, upper_bound, solution_time, incumbent_count, hydraulic_feasible_count = operational_problem_two_phased_soc(results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit, valid_inequalities)
                elif method == 'gurobi_callback':
                    define_soc_constraints(pipes, pumps, valves, time_steps, cone_budget)
                    status, solution, lower_bound, upper_bound, solution_time, incumbent_count, hydraulic_feasible_count = operational_problem_soc_callback(results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit, valid_inequalities)
                else:
                    status, solution, lower_bound, upper_bound, solution_time = PyomoOperationalModel(method, method, results_directory, junctions, tanks, pipes, pumps, valves, time_steps, hydraulic_timestep, time_limit)

                write_results(results_folder, 'results.txt', f'{epanet_model} | {method} | {status} | {f"{lower_bound:.2f}" if lower_bound is not None else "-"} | {f"{upper_bound:.2f}" if upper_bound is not None else "-"} | {f"{incumbent_count}" if incumbent_count else "-"} | {f"{hydraulic_feasible_count}" if hydraulic_feasible_count else "-"} | {solution_time:.2f} | {f"{obbt_time:.2f}" if obbt_flag else "-"} | {f"{hull_facet_time:.2f}" if hull_facet_time else "-"}')